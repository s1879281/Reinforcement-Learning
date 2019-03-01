#!/usr/bin/env python3
# encoding utf-8

import argparse
import random
from collections import defaultdict, OrderedDict

from DiscreteHFO.Agent import Agent
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer


class MonteCarloAgent(Agent):
    def __init__(self, discountFactor, epsilon, initVals=0.0):
        super(MonteCarloAgent, self).__init__()
        self.discountFactor = discountFactor
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.initVals = initVals
        self.QValueTable = defaultdict(lambda: initVals)
        self.returnsDict = defaultdict(lambda: [])
        self.stateActionList = []
        self.rewardList = []
        self.statusList = []

    def learn(self):
        G = 0
        GList = []
        tempDict = OrderedDict()
        curEpisodeList = []

        for i in range(1, len(self.rewardList) + 1):
            G = self.discountFactor * G + self.rewardList[-i]
            GList.insert(0, G)

        for stateAction, GValue in zip(self.stateActionList, GList):
            if stateAction not in tempDict.keys():
                tempDict[stateAction] = GValue

        for stateAction, GValue in tempDict.items():
            self.returnsDict[stateAction].append(GValue)
            self.QValueTable[stateAction] = sum(self.returnsDict[stateAction]) / len(self.returnsDict[stateAction])
            curEpisodeList.append(self.QValueTable[stateAction])

        return self.QValueTable, curEpisodeList

    def toStateRepresentation(self, state):
        state = str(state)
        if (state, 'KICK') not in self.QValueTable.keys():
            self.QValueTable.update({(state, action): self.initVals for action in self.possibleActions})

        return state

    def setExperience(self, state, action, reward, status, nextState):
        self.stateActionList.append((state, action))
        self.rewardList.append(reward)
        self.statusList.append(status)

    def setState(self, state):
        self.curState = state

    def reset(self):
        self.stateActionList = []
        self.rewardList = []
        self.statusList = []

    def act(self):
        randomNum = random.random()
        if randomNum < self.epsilon:
            return random.choice(self.possibleActions)
        else:
            actionDict = {key[1]: value for key, value in self.QValueTable.items() if key[0] == self.curState}
            return random.choice(
                [action for action, value in actionDict.items() if value == max(actionDict.values())])

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        epsilon = self.initEpsilon * 0.85 ** (episodeNumber // 100)

        return epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args = parser.parse_args()

    # Init Connections to HFO Server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
    hfoEnv.connectToServer()

    # Initialize a Monte-Carlo Agent
    agent = MonteCarloAgent(discountFactor=0.99, epsilon=1.0)
    numEpisodes = args.numEpisodes
    numTakenActions = 0
    # Run training Monte Carlo Method
    for episode in range(numEpisodes):
        agent.reset()
        observation = hfoEnv.reset()
        status = 0

        while status == 0:
            epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            observation = nextObservation

        agent.learn()
