import logging
import gym
from dxc_env import register_xac_env
from compiler_gym.util.logging import init_logging

def main():
    init_logging(level=logging.DEBUG)
    register_xac_env()
    with gym.make("xac-v0", observation_space="isa_metrics", reward_space="occupancy") as env:
        env.reset()
        i = 0
        for _ in range(10000):
            observation, reward, done, info = env.step(env.action_space.sample())
            i += 1
            if done:
                env.reset()
                break

        print(f"Total steps: {i}")

if __name__ == '__main__':
    main()