import logging
import gym
from dxc_env import register_dxc_env
from compiler_gym.util.logging import init_logging

def main():
    init_logging(level=logging.DEBUG)
    register_dxc_env()
    with gym.make("dxc-v0") as env:
        env.reset()
        for _ in range(1):
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()

if __name__ == '__main__':
    main()