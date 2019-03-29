#!/usr/bin/env python3
# encoding utf-8

import argparse
from copy import deepcopy
from collections import defaultdict
from DiscreteMARLUtils.Agent import Agent
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
import random
import numpy as np


class WolfPHCAgent(Agent):
    def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
        super(WolfPHCAgent, self).__init__()
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.winDelta = winDelta
        self.loseDelta = loseDelta
        self.initVals = initVals
        self.QValueTable = defaultdict(lambda: initVals)
        self.policyTable = defaultdict(lambda: [])
        self.avgPolicyTable = defaultdict(lambda: [])
        self.C = defaultdict(lambda: initVals)


    def setExperience(self, state, action, reward, status, nextState):
        self.state = state
        self.action = action
        self.reward = reward
        self.status = status
        self.nextState = nextState
        if (nextState, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(nextState, action): self.initVals for action in self.possibleActions})
            self.policyTable.update({nextState: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})
            self.avgPolicyTable.update({nextState: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})

    def learn(self):
        nextStateQList = [value for key, value in self.QValueTable.items() if key[0] == self.nextState]
        update = self.learningRate * (
                self.reward + self.discountFactor * max(nextStateQList) - self.QValueTable[
            (self.curState, self.action)])
        self.QValueTable[(self.curState, self.action)] += update

        return update

    def act(self):
        probList = self.policyTable[self.curState]
        action = np.random.choice(self.possibleActions, p=probList)

        return action

    def calculateAveragePolicyUpdate(self):
        self.C[self.curState] += 1
        update = 1 / self.C[self.curState] * (np.array(self.policyTable[self.curState]) - np.array(self.avgPolicyTable[self.curState]))
        self.avgPolicyTable[self.curState] += update
        self.avgPolicyTable[self.curState] = list(self.avgPolicyTable[self.curState])

        return self.avgPolicyTable[self.curState]

    def calculatePolicyUpdate(self):
        actionDict = {key[1]: value for key, value in self.QValueTable.items() if key[0] == self.curState}
        AsubOptimal = [action for action, value in actionDict.items() if value != max(actionDict.values())]
        sumPolicy = 0.
        sumAvgPolicy = 0.
        for i, action in enumerate(self.possibleActions):
            sumPolicy += self.policyTable[self.curState][i] * self.QValueTable[(self.curState, action)]
            sumAvgPolicy += self.avgPolicyTable[self.curState][i] * self.QValueTable[(self.curState, action)]
        if sumPolicy >= sumAvgPolicy:
            delta = self.winDelta
        else:
            delta = self.loseDelta
        pMoved = 0.
        for i, action in enumerate(self.possibleActions):
            if action in AsubOptimal:
                pMoved += min(delta / len(AsubOptimal), self.policyTable[self.curState][i])
                self.policyTable[self.curState][i] -= min(delta / len(AsubOptimal), self.policyTable[self.curState][i])
        for i, action in enumerate(self.possibleActions):
            if action not in AsubOptimal:
                self.policyTable[self.curState][i] += pMoved / (6 - len(AsubOptimal))

        return self.policyTable[self.curState]

    def toStateRepresentation(self, state):
        state = str(state)

        return state

    def setState(self, state):
        if (state, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(state, action): self.initVals for action in self.possibleActions})
            self.policyTable.update({state: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})
            self.avgPolicyTable.update({state: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]})
        self.curState = state

    def setLearningRate(self, lr):
        self.learningRate = lr

    def setWinDelta(self, winDelta):
        self.winDelta = winDelta

    def setLoseDelta(self, loseDelta):
        self.loseDelta = loseDelta

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        learningRate = max(0.2 * 0.95 ** (episodeNumber // 100), 0.05)
        winDelta = 0.01
        loseDelta = 0.1

        return loseDelta, winDelta, learningRate


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=100000)

    args = parser.parse_args()

    numOpponents = args.numOpponents
    numAgents = args.numAgents
    MARLEnv = DiscreteMARLEnvironment(numOpponents=numOpponents, numAgents=numAgents)

    agents = []
    for i in range(args.numAgents):
        agent = WolfPHCAgent(learningRate=0.2, discountFactor=0.99, winDelta=0.01, loseDelta=0.1)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME", "IN_GAME", "IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0

        while status[0] == "IN_GAME":
            for agent in agents:
                loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
                agent.setLoseDelta(loseDelta)
                agent.setWinDelta(winDelta)
                agent.setLearningRate(learningRate)
            actions = []
            perAgentObs = []
            agentIdx = 0
            for agent in agents:
                obsCopy = deepcopy(observation[agentIdx])
                perAgentObs.append(obsCopy)
                agent.setState(agent.toStateRepresentation(obsCopy))
                actions.append(agent.act())
                agentIdx += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1

            agentIdx = 0
            for agent in agents:
                agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx],
                                    reward[agentIdx],
                                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agent.learn()
                agent.calculateAveragePolicyUpdate()
                agent.calculatePolicyUpdate()
                agentIdx += 1

            totalReward += sum(reward)
            observation = nextObservation

        if totalReward != 0:
            print('episode', episode)
            print('numTakenActions', numTakenActions)
            print('total_reward', totalReward)
