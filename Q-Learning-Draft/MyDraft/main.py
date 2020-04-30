from maze_env import Maze
from RL_Brain import RL_Brain
import numpy as np

def run_maze():
    step = 0
    for episode in range(100000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.get_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_observation(observation, action, reward, observation_)

            # if (step > 200) and (step % 5 == 0):
            RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = RL_Brain(env.n_features, env.n_actions)
    env.after(100, run_maze)
    env.mainloop()