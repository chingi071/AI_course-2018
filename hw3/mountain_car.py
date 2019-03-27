
# coding: utf-8

# In[11]:


import gym
import math
import numpy as np


def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon:  # random select an action with probability epsilon
        return action_space.sample()  # random select an action
    else:  # Choose an action according to the Q table
        return np.argmax(q_table[state])


def get_state(observation, n_buckets):
    state = [0] * len(observation)
    state_bounds = [(-1.2,0.6),(-0.07,0.07)]
    
    #position
    p = (state_bounds[0][1] - state_bounds[0][0]) /n_buckets[0]

    #velocity
    v = (state_bounds[1][1] - state_bounds[1][0]) /n_buckets[1]
    
    s1 = (observation[0] -state_bounds[0][0]) / p
    s1 = math.ceil(s1)
    s2 = (observation[1] -state_bounds[1][0]) / v
    s2 = math.ceil(s2)
    
    state[0] = state[0] + s1 - 1
    state[1] = state[1] + s2 - 1
    
#     print(observation, state)
    
    return tuple(state)


def init_qtable():
    n_buckets = (10, 10) 
    n_actions = env.action_space.n  # Number of actions
    q_table = np.zeros(n_buckets + (n_actions,))

    return n_buckets, n_actions, q_table


def get_epsilon(i):
    epsilon = 1
    epsilon_min = 0.005
    epsilon_decay = (epsilon - epsilon_min) / 100*(i+1)
    epsilon -= epsilon_decay
    return epsilon                                          


def get_lr(i):
    lr = 0.1
    return lr


def get_gamma(i):
    gamma = 0.99
    return gamma

# Main
env = gym.make("MountainCar-v0")  # Select an environment


sum_rewards = 0
n_success = 0
f = open('mountaincar_qtable_log.txt', 'w')

n_buckets, n_actions, q_table = init_qtable()

# Q-learning
for i_episode in range(1000):
    total_reward = 0
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    observation = env.reset()  # reset the environment and return the default observation
    state = get_state(observation, n_buckets)

    for t in range(1000):
        #env.render()  # Show the environment in GUI

        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, info = env.step(action)  # Take an action on current environment
        total_reward += reward
        sum_rewards += reward

        next_state = get_state(observation, n_buckets)

        q_next_max = np.amax(q_table[next_state])  # The expected max reward for next state
        q_table[state + (action,)] += lr * (reward + get_gamma(i_episode) *
                                            q_next_max - q_table[state + (action,)])  # Update Q table
        state = next_state
        if done:  # Episode terminate
            break

    if total_reward > -200:
        n_success += 1
    if (i_episode + 1) % 100 == 0:
        avg_rewards = sum_rewards / (i_episode + 1)
        log = '{:<2d} successes in {:4d} episodes, avg = {}'.format(n_success, i_episode + 1, avg_rewards)
        print(log)
        f.write(log + '\n')

#     print('Episode {:4d}, total rewards {}'.format(i_episode + 1, total_reward))

f.close()
env.close()

