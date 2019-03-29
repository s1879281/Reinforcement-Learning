#!/usr/bin/env python3
# encoding utf-8

from collections import defaultdict
import argparse
from copy import deepcopy
import random
import numpy as np

from DiscreteMARLUtils.Agent import Agent
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment


class IndependentQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(IndependentQLearningAgent, self).__init__()
        self.initLearningRate = learningRate
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.initVals = initVals
        self.QValueTable = defaultdict(lambda: initVals)

    def setExperience(self, state, action, reward, status, nextState):
        self.state = state
        self.action = action
        self.reward = reward
        self.status = status
        self.nextState = nextState
        if (nextState, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(nextState, action): self.initVals for action in self.possibleActions})

    def learn(self):
        nextStateQList = [value for key, value in self.QValueTable.items() if key[0] == self.nextState]
        update = self.learningRate * (
                self.reward + self.discountFactor * max(nextStateQList) - self.QValueTable[
            (self.curState, self.action)])
        self.QValueTable[(self.curState, self.action)] += update

        return update

    def act(self):
        randomNum = random.random()
        if randomNum < self.epsilon:
            return random.choice(self.possibleActions)
        else:
            actionDict = {key[1]: value for key, value in self.QValueTable.items() if key[0] == self.curState}
            return random.choice(
                [action for action, value in actionDict.items() if value == max(actionDict.values())])

    def toStateRepresentation(self, state):
        state = str(state)

        return state

    def setState(self, state):
        if (state, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(state, action): self.initVals for action in self.possibleActions})
        self.curState = state

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        learningRate = max(0.5 * 0.95 ** (episodeNumber // 100), 0.05)
        epsilon = 1. * ((1 - 1 / (1 + np.exp(-numTakenActions / 250))) * 2 * 0.9 + 0.1)

        return learningRate, epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args = parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents=args.numOpponents, numAgents=args.numAgents)
    agents = []
    for i in range(args.numAgents):
        agent = IndependentQLearningAgent(learningRate=0.1, discountFactor=0.9, epsilon=1.0)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME", "IN_GAME", "IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0

        while status[0] == "IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(args.numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx],
                                               reward[agentIdx],
                                               status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            totalReward += sum(reward)

            observation = nextObservation

        if totalReward != 0:
            print('episode',episode)
            print('numTakenActions',numTakenActions)
            print('total_reward',totalReward)
