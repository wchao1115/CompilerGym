import logging
import gym
from xac_env import register_xac_env
from xac_logger import init_logger

logger = init_logger("xac_cli")

class Trainer():
    def __init__(self, env: gym.Env):
        self._env = env
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self._reward_space = env.reward_space

    def fit(self, num_episodes: int=100):
        benchmarks = iter(self._env.datasets.benchmarks())
        episode = 0
        total_iterations = 0

        while episode < num_episodes:
            benchmark = next(benchmarks, None)
            if benchmark is None:
                break

            self._env.reset(benchmark)
            done = False
            total_reward = 0
            iterations = 0

            while not done:
                action = self._action_space.sample()
                observation, reward, done, info = self._env.step(action)
                iterations += 1
                total_reward += reward

            logger.info(f"Episode {episode + 1}/{num_episodes}, Iterations: {iterations}, Total Reward: {total_reward}")
            episode += 1
            total_iterations += iterations
            
        logger.info(f"Training completed. Total Iterations: {total_iterations}")

def main():
    register_xac_env()
    trainer = Trainer(env=gym.make("xac-v0", observation_space="isa_metrics", reward_space="occupancy"))
    trainer.fit(num_episodes=30)

if __name__ == '__main__':
    main()