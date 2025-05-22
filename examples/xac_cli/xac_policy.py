import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple
from collections import namedtuple
from xac_utils import MovingExponentialAverage, or_mask_across_episodes

ActionValue = namedtuple("ActionValue", ["log_prob", "value"])

class Policy(nn.Module):
    """An actor-critic policy."""
    
    mean_smoothing = 0.95   # Smoothing factor for mean normalization
    std_smoothing = 0.4     # Smoothing factor for std dev normalization

    eps = np.finfo(np.float32).eps.item()

    def __init__(
            self, 
            episode_length: int, 
            num_actions: int, 
            hidden_size: int
        ):
        super().__init__()
        self._fc1 = nn.Linear(episode_length * num_actions, hidden_size)
        self._fc2 = nn.Linear(hidden_size, hidden_size)
        self._fc3 = nn.Linear(hidden_size, hidden_size)
        self._fc4 = nn.Linear(hidden_size, hidden_size)

        # Actor's output layer
        self._actor_head = nn.Linear(hidden_size, num_actions)

        # Critic's output layer
        self._critic_head = nn.Linear(hidden_size, 1)

        # Action values and rewards throughout the episode
        self._action_values: List[ActionValue] = []
        self._rewards: List[float] = []

        # Keep exponential moving average of mean and std to normalize the input
        self._moving_mean = MovingExponentialAverage(self.mean_smoothing)
        self._moving_std = MovingExponentialAverage(self.std_smoothing)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward of both actor and critic"""

        # Input layer maps the input state (x), which is a sequence of one-hot vectors, into a vector
        # of the hidden size, the next layers retain the same size and use residual connections
        x = F.relu(self._fc1(x))
        x = x.add(F.relu(self._fc2(x)))
        x = x.add(F.relu(self._fc3(x)))
        x = x.add(F.relu(self._fc4(x)))

        # Actor maps the input state to logits of each action
        logits = self._actor_head(x)

        # Critic evaluates the input state into a single value
        state_value = self._critic_head(x)

        # Return a tuple of 2 values:
        # 1. a list with the logits of each action over the action space
        # 2. the value of the input state
        return logits, state_value
    
    def select_action(self, state, mask, temperature: float = 1.0, exploration_rate: float = 0) -> int:
        """Selects an action from the input state and save it for the record"""

        logits, value = self(state)

        if mask is not None:
            # mask off actions already taken
            logits = logits.masked_fill(mask, float("-inf"))

        # optional temperature scaling
        logits = logits / temperature

        # create a probability distribution where the probability of action i is probs[i]
        distribution = Categorical(logits=logits)

        # sample an action from the distribution or pick an action randomly if in an exploration mode
        if random.random() < exploration_rate:
            # only sample from valid (unmasked) actions
            valid_actions = (~mask).nonzero(as_tuple=True)[0] if mask is not None else torch.arange(len(logits))
            action = valid_actions[torch.randint(0, len(valid_actions), ())]
        else:
            action = distribution.sample()

        # Storing both the log probability and the state value together in an ActionValue object 
        # makes it easy to later compute losses and perform updates during training.Recording this 
        # info ensures that all relevant information about actions taken during an episode is collected 
        # and can be used for learning after the episode ends. 
        log_prob = distribution.log_prob(action)
        self._action_values.append(ActionValue(log_prob, value))
        return action.item()

    def finish_episode(self, optimizer) -> float:
        """Calculates actor and critic loss and performs backprop."""

        reward_sofar = 0
        returns = []  # list to save the true values

        # Computing the returns, which are cumulative sums of rewards collected during the episode. This is done 
        # by iterating over the rewards in reverse order and inserting the running total at the front of the returns 
        # list. This approach ensures that each entry in returns represents the sum of all rewards from that timestep 
        # onward. No discount factor is used, which is appropriate for short, fixed-length episodes.
        for reward in self._rewards[::-1]:
            reward_sofar += reward
            returns.insert(0, reward_sofar)

        # The returns are converted to a PyTorch tensor and normalized using moving averages of the mean and standard deviation. 
        # This normalization helps stabilize training by keeping the scale of the returns consistent across episodes.
        returns = torch.tensor(returns)
        self._moving_mean.next(returns.mean())
        self._moving_std.next(returns.std())        
        returns = (returns - self._moving_mean.value) / (self._moving_std.value + self.eps)

        policy_losses = []
        value_losses = []

        # Then, iterates over pairs of saved action values (log probabilities and value estimates) and the corresponding 
        # normalized returns. For each pair, it computes the "advantage," which measures how much better the outcome was 
        # compared to what the critic predicted. The policy loss is calculated as the negative log probability of the action, 
        # weighted by the advantage. This encourages the model to increase the probability of actions that led to better-than-expected 
        # outcomes and decrease it for worse-than-expected ones. The critic loss is computed using the smooth L1 loss 
        # between the predicted value and the actual return.
        for (log_prob, value), reward_sofar in zip(self._action_values, returns):

            # The advantage is how much better a situation turned out in
            # this case than the critic expected it to.
            advantage = reward_sofar - value.item()

            # Calculate the actor (policy) loss. The log_prob is symbolic, so during backpropagation:
            # - If the advantage is positive, the probability of taking the chosen action will be increased.
            # - If the advantage is negative, the probability will be decreased.
            # This allows the model to learn a probability distribution over actions, even though we cannot
            # directly backpropagate through the sampling process itself.
            #
            # Note: If the critic becomes perfectly accurate (so the advantage is always zero), the policy
            # would stop learning because the loss gradient would be zero. However, since the critic can only
            # predict the expected value for a state (not for each possible action), there will always be some
            # actions that are better or worse than average. As long as the policy assigns non-zero probability
            # to multiple actions, the critic will sometimes be wrong, and learning will continue.
            policy_losses.append(-log_prob * advantage)

            # Calculate critic (value) loss using L1 smooth loss.
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward_sofar])))

        # Reset gradients
        optimizer.zero_grad()

        # Sum up all the values of policy_losses and value_losses to get the total loss.
        # Perform backpropagation to compute gradients and update the model parameters.
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        # Reset rewards and action buffer.
        del self._rewards[:]
        del self._action_values[:]

        return loss_value
    
    def append_reward(self, reward: float):
        self._rewards.append(reward)
