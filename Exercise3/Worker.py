import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np
from hfo import *

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
    port = 6000 + idx
    seed = 123 + idx * 10
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    thread_counter = 0
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    target_value_network.load_state_dict(torch.load('./checkpoint.pth'))
    num_episode = 0
    goal_list = []

    while True:
        epsilon = args.epsilon * ((1 - 1 / (1 + np.exp(-thread_counter / 250))) * 2 * 0.9 + 0.1)
        print('epsilon\n\n\n', epsilon)
        print('episode\n\n\n', num_episode)
        done = False
        curState = hfoEnv.reset()
        timestep = 0

        while not done and timestep < 500:
            action = epsilon_greedy(curState, epsilon, value_network)
            nextState, reward, done, status, info = hfoEnv.step(hfoEnv.possibleActions[action])

            pred_value = computePrediction(curState, action, value_network)
            target_value = computeTargets(reward, nextState, args.discountFactor, done, target_value_network)
            loss = criterion(pred_value, target_value)
            loss.backward()

            curState = nextState

            with lock:
                counter.value += 1
            thread_counter += 1
            timestep += 1

            if status == GOAL:
                goal_list.append(num_episode)

            if counter.value % args.iterate_target == 0:
                target_value_network.load_state_dict(torch.load('./checkpoint.pth'))


            if thread_counter % args.iterate_async == 0 or done:
                optimizer.step()
                optimizer.zero_grad()
                saveModelNetwork(value_network, './checkpoint.pth')

        num_episode += 1


def epsilon_greedy(state, epsilon, value_network):
    randomNum = random.random()
    if randomNum < epsilon:
        return random.choice(list(range(4)))
    else:
        return torch.argmax(value_network(state)[0]).item()


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    if done:
        return torch.tensor(reward)
    else:
        target_value = reward + discountFactor * max(targetNetwork(nextObservation)[0])
        return target_value


def computePrediction(state, action, valueNetwork):
    pred_value = valueNetwork(state)[0][action]
    return pred_value

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)





