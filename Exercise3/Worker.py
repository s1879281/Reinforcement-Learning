import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
    port = 6000 + idx
    seed = 123 + idx * 10
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    thread_counter = 0
    criterion = nn.MSELoss()
    optimizer.zero_grad()

    while counter <= args.num_episodes:
        done = False
        curState = hfoEnv.reset()

        while not done:
            action = epsilon_greedy(curState, args.epsilon, value_network)
            nextState, reward, done, status, info = hfoEnv.step(hfoEnv.possibleActions[action])
            curState = nextState

            pred_value = computePrediction(curState, action, value_network)
            target_value = computeTargets(reward, nextState, args.discountFactor, done, target_value_network)
            loss = criterion(pred_value, target_value)
            loss.backward()

            counter += 1
            thread_counter += 1

            if counter % args.iterate_target == 0:
                lock.acquire()
                target_value_network.load_state_dict(torch.load('./checkpoint.pth'))
                lock.release()


            if thread_counter % args.iterate_async == 0 or done:
                optimizer.step()
                optimizer.zero_grad()
                lock.acquire()
                saveModelNetwork(value_network, './checkpoint.pth')
                lock.release()


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





