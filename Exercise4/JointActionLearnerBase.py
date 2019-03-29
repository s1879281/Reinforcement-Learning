#!/usr/bin/env python3
# encoding utf-8

import argparse
from collections import defaultdict
from copy import deepcopy
import random
import numpy as np

from DiscreteMARLUtils.Agent import Agent
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment


class JointQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
        super(JointQLearningAgent, self).__init__()
        self.initLearningRate = learningRate
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.initVals = initVals
        self.QValueTable = defaultdict(lambda: initVals)
        self.C = defaultdict(lambda: initVals)
        self.n = defaultdict(lambda: initVals)

    def setExperience(self, state, action, oppoActions, reward, status, nextState):
        self.state = state
        self.action = action
        self.oppoAction = oppoActions[0]
        self.reward = reward
        self.status = status
        self.nextState = nextState
        if (nextState, ('KICK', 'KICK')) not in self.QValueTable.keys():
            self.QValueTable.update({(nextState, (action, oppoAction)): self.initVals
                for action in self.possibleActions for oppoAction in self.possibleActions})

        self.C[self.state, self.oppoAction] += 1
        self.n[self.state] += 1

    def learn(self):
        actionPairDict = {key[1]: value for key, value in self.QValueTable.items()
                          if key[0] == self.nextState}
        if self.n[self.nextState] != 0:
            actionPairDict = {actionPair: value * self.C[self.nextState, actionPair[1]] / self.n[self.nextState]
                              for actionPair, value in actionPairDict.items()}
        actionDict = {action: max([value for actionPair, value in actionPairDict.items() if actionPair[0] == action])
                      for action in self.possibleActions}
        VnextState = max(actionDict.values())
        update = self.learningRate * (
                self.reward + self.discountFactor * VnextState - self.QValueTable[
            self.curState, (self.action, self.oppoAction)])
        self.QValueTable[self.curState, (self.action, self.oppoAction)] += update

        return update

    def act(self):
        actionPairDict = {key[1]: value for key, value in self.QValueTable.items()
                               if key[0] == self.curState}
        if self.n[self.curState] != 0:
            actionPairDict = {actionPair: (value * self.C[self.curState, actionPair[1]] / self.n[self.curState])
                                   for actionPair, value in actionPairDict.items()}
        actionDict = {action: max([value for actionPair, value in actionPairDict.items() if actionPair[0] == action])
                      for action in self.possibleActions}

        return random.choice(
                [action for action, value in actionDict.items() if value == max(actionDict.values())])

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setState(self, state):
        if (state, ('KICK', 'KICK')) not in self.QValueTable.keys():
            self.QValueTable.update({(state, (action, oppoAction)): self.initVals
                for action in self.possibleActions for oppoAction in self.possibleActions})
        self.curState = state

    def toStateRepresentation(self, rawState):
        state = str(rawState)

        return state

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        learningRate = max(0.5 * 0.95 ** (episodeNumber // 100), 0.05)
        epsilon = 1. * ((1 - 1 / (1 + np.exp(-numTakenActions / 500))) * 2 * 0.9 + 0.1)

        return learningRate, epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args = parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents=args.numOpponents, numAgents=args.numAgents)
    agents = []
    numAgents = args.numAgents
    numEpisodes = args.numEpisodes
    for i in range(numAgents):
        agent = JointQLearningAgent(learningRate=0.1, discountFactor=0.9, epsilon=1.0, numTeammates=args.numAgents - 1)
        agents.append(agent)

    numEpisodes = numEpisodes
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
                agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())

            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            for agentIdx in range(args.numAgents):
                oppoActions = actions.copy()
                del oppoActions[agentIdx]
                agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]),
                                               actions[agentIdx], oppoActions,
                                               reward[agentIdx], status[agentIdx],
                                               agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            totalReward += sum(reward)
            observation = nextObservation

        if totalReward != 0:
            print('episode', episode)
            print('numTakenActions', numTakenActions)
            print('total_reward', totalReward)
