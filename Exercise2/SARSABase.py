#!/usr/bin/env python3
# encoding utf-8

import argparse
import random
from collections import defaultdict

import numpy as np

from DiscreteHFO.Agent import Agent
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer


class SARSAAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(SARSAAgent, self).__init__()
        self.initLearningRate = learningRate
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.initVals = initVals
        self.QValueTable = defaultdict(lambda: initVals)
        self.stateList = []
        self.actionList = []
        self.rewardList = []
        self.statusList = []

    def learn(self):
        update = self.learningRate * (
                self.rewardList[-2] + self.discountFactor * self.QValueTable[
            (self.stateList[-1], self.actionList[-1])] - self.QValueTable[(self.stateList[-2], self.actionList[-2])])
        self.QValueTable[(self.stateList[-2], self.actionList[-2])] += update

        return update

    def act(self):
        randomNum = random.random()
        if randomNum < self.epsilon:
            return random.choice(self.possibleActions)
        else:
            actionDict = {key[1]: value for key, value in self.QValueTable.items() if key[0] == self.curState}
            return random.choice(
                [action for action, value in actionDict.items() if value == max(actionDict.values())])

    def setState(self, state):
        self.curState = state
        if (self.curState, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(self.curState, action): self.initVals for action in self.possibleActions})

    def setExperience(self, state, action, reward, status, nextState):
        self.stateList.append(state)
        self.actionList.append(action)
        self.rewardList.append(reward)
        self.statusList.append(status)

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        learningRate = self.initLearningRate * 0.95 ** (episodeNumber // 100)
        epsilon = self.initEpsilon * ((1 - 1 / (1 + np.exp(-numTakenActions / 250))) * 2 * 0.9 + 0.1)

        return learningRate, epsilon

    def toStateRepresentation(self, state):
        return str(state)

    def reset(self):
        self.stateList = []
        self.actionList = []
        self.rewardList = []
        self.statusList = []

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args = parser.parse_args()

    numEpisodes = args.numEpisodes
    # Initialize connection to the HFO environment using HFOAttackingPlayer
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
    hfoEnv.connectToServer()

    # Initialize a SARSA Agent
    agent = SARSAAgent(0.1, 0.99, 1.0)

    # Run training using SARSA
    numTakenActions = 0
    for episode in range(numEpisodes):
        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True

        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            print(learningRate,epsilon)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))

            if not epsStart:
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()
