import logging
import gym
import torch
import torch.optim as optim
import numpy as np
from typing import cast
from torch.distributions import Categorical
from xac_env import register_xac_env, XacDataset
from xac_logger import init_logger
from xac_policy import Policy
from xac_utils import MovingExponentialAverage, or_mask_across_episodes, round_robin_iter, sequential_iter

logger = init_logger("xac_cli")

class Learner():

    learning_rate = 0.001 # Learning rate for the optimizer
    episode_length = 50 # maximum number of attempts different actions are tried in an episode, must match xac/server/dxc_service.py
    split_ratio = 0.8 # proportion of the dataset used for training vs. testing
    exploration_rate_init = 1.0 # Initial exploration rate for action selection
    exploration_rate_decay = 0.99 # Decay rate for exploration, so it's less random over time
    
    def __init__(self, env: gym.Env):
        self._env = env
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self._reward_space = env.reward_space
        self._policy = Policy(episode_length=self.episode_length, num_actions=self._action_space.n, hidden_size=128)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self.learning_rate)
        self._exploration_rate = self.exploration_rate_init

    def fit(self, num_episodes: int = 100):
        """ Trains the policy over multiple episodes, updating exploration rate and logging statistics after each episode."""

        self._policy.train()

        episode = 0
        total_steps = 0
        avg_reward = MovingExponentialAverage(0.95)
        avg_loss = MovingExponentialAverage(0.95)

        training_set = round_robin_iter(list(self._env.datasets.benchmarks()))

        while episode < num_episodes:
            benchmark = next(training_set, None)
            if benchmark is None:
                break

            state = self._env.reset(benchmark, observation_space="states")
            done = False
            total_reward = 0
            steps = 0

            while not done and steps < self.episode_length:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                
                action = self._policy.select_action(
                    state_tensor,
                    # mask any action already taken in the previous steps
                    or_mask_across_episodes(
                        mask, self.episode_length, self._action_space.n, steps
                    ) if steps > 0 else None,
                    exploration_rate=self._exploration_rate
                )
                state, reward, done, _ = self._env.step(action)
                mask = torch.tensor(state == 1, dtype=torch.bool)

                self._policy.append_reward(reward)
                total_reward += reward
                steps += 1

            # train the policy with what happened during the episode
            loss = self._policy.finish_episode(self._optimizer)

            # update the exploration rate so action selection is less random over time
            self._exploration_rate = max(0.1, self._exploration_rate * self.exploration_rate_decay)

            # stats gathering
            avg_reward.next(total_reward)
            avg_loss.next(loss)
            episode += 1
            total_steps += steps

            logger.info(
                f"Episode [{episode}/{num_episodes}] "
                f"Steps: {steps}, " 
                f"Total Reward: {total_reward}, "
                f"Avg Reward: {avg_reward.value:.2f}, "
                f"Avg Loss: {avg_loss.value:.2f}, "
                f"Epsilon: {self._exploration_rate:.4f}"
            )

        logger.info(f"Training completed. Total attempts: {total_steps}")

    def predict(self, benchmark, temperature: float = 1.0, greedy: bool = True):
        """ Run the trained policy on a given benchmark and return the sequence of actions and total reward."""

        self._policy.eval()

        state = self._env.reset(benchmark, observation_space="states")
        done = False
        actions = []
        total_reward = 0
        steps = 0

        with torch.no_grad():
            while not done and steps < self.episode_length:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                logits, _ = self._policy(state_tensor)

                if steps > 0:
                    # mask off actions already taken from the previous steps
                    logits = logits.masked_fill(
                        or_mask_across_episodes(
                            mask, self.episode_length, self._action_space.n, steps
                        ),
                        float("-inf")
                    )

                # optional temperature scaling
                logits = logits / temperature

                if greedy:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    action = Categorical(logits=logits).sample().item()

                state, reward, done, _ = self._env.step(action)
                mask = torch.tensor(state == 1, dtype=torch.bool)

                actions.append(action)
                total_reward += reward
                steps += 1
                """
                logger.info(
                    f"Step: [{steps}/{max_steps}] "
                    f"State: {np.sum(state_tensor.numpy())}, "
                    f"Action: {action}, "
                    f"Reward: {reward} "
                )
                """

        return actions, total_reward
    
    def cv(self, greedy: bool = True):
        testing_set = sequential_iter(list(self._env.datasets.benchmarks()))
        improvements = []

        while True:
            benchmark = next(testing_set, None)
            if benchmark is None:
                break

            actions, _ = self.predict(benchmark, greedy=greedy)

            learned = self._env.reward_space.last_occupancy
            base_metrics = self._env.reset(benchmark, observation_space="metrics")
            baseline = self._env.reward_space.get_occupancy([base_metrics])

            improvements += [(learned - baseline) / baseline]
            logger.info(f"{benchmark.uri}: Baseline: {baseline}, Learned: {learned}, Steps: {len(actions)}")

        logger.info(f"Cross-validation completed Avg improvements (%): {(np.mean(improvements) * 100):.2f}, Greedy: {greedy}")

def main():
    register_xac_env()
    learner = Learner(env=gym.make("xac-v0"))
    learner.fit(num_episodes=50)
    learner.cv()

if __name__ == '__main__':
    main()