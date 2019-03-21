import gym
import numpy as np
import math

env = gym.make('CartPole-v0')

NUM_STATE = 10
NUM_ACTION = env.action_space.n
ANGLE_STATE_INDEX = 2
MIN_EXPLORE_RATE = 0.1
MIN_LEARNING_RATE = 0.1

NUM_EPISODES = 1000
MAX_T = 500


buckets = (1, 1, 6, 12,)
q_table = np.zeros(buckets + (NUM_ACTION,))


def simulate(render):
    learning_rate = 0.2
    explore_rate = 0.8
    discount_factor = 1

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        state_0 = discretize(obs)
        t_done = 0
        for t in range(MAX_T):
            if (render):
                env.render()
            action = select_action(state_0, explore_rate)
            obs, reward, done, info = env.step(action)
            state = discretize(obs)
            q_table[state_0][action] += learning_rate * \
                (reward + discount_factor *
                 np.max(q_table[state]) - q_table[state_0][action])

            state_0 = state
            if not done:
                t_done = t
            if done:
                print("Episode %d finished after %f time steps" %
                      (episode, t_done))
                if (t_done > 195):
                    print("Ganhou!")
                break

        print("Explore rate: %f, Learning rate: %f"
              % (explore_rate, learning_rate))
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    if np.random.uniform(0, 1) < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action


def discretize(obs):
    upper_bounds = [env.observation_space.high[0],
                    0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0],
                    -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [obs[i] + abs(lower_bounds[i]) / (upper_bounds[i] -
                                               lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i]))
               for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i]))
               for i in range(len(obs))]
    return tuple(new_obs)


def get_explore_rate(t):
    return min(1, MIN_EXPLORE_RATE*NUM_EPISODES/(t+1))


def get_learning_rate(t):
    return min(0.8, MIN_LEARNING_RATE*NUM_EPISODES/(t+1))


if __name__ == "__main__":
    simulate(False)
