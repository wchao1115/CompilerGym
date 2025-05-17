import logging
import gym
from dxc_env import register_xac_env
from compiler_gym.util.logging import init_logging

def main():
    init_logging(level=logging.DEBUG)
    register_xac_env()
    with gym.make("xac-v0", observation_space="isa_metrics", reward_space="occupancy") as env:
        env.reset()
        for _ in range(1):
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()

if __name__ == '__main__':
    main()