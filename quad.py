from agent import Agent
from task import Task
import numpy as np
import sys

num_episodes = 1000
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)
agent = Agent(task) 

best_score = -1000.
for i_episode in range(1, num_episodes+1):
    state = agent.reset_state() # start a new episode
    while True:
        _, next_state, reward, done = agent.act(state) 
        #next_state, reward, done = task.step(action)
        #agent.step(reward, done)
        state = next_state
        if done:
            best_score = max(best_score, task.total_rewards)
            #print("\rEpisode = {:4d}, score = {:7.3f}, best={:7.3f}".format(i_episode, task.total_rewards, best_score), end="")  # [debug]
            print('{"metric": "Score", "value": ', task.total_rewards, '}')
            print('{"metric": "Episode", "value": ', i_episode, '}')
            break
    #sys.stdout.flush()