#!/usr/bin/env python3
# encoding utf-8

import argparse
import random
from collections import defaultdict

from DiscreteHFO.Agent import Agent
from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer


class QLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(QLearningAgent, self).__init__()
        self.initLearningRate = learningRate
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.initEpsilon = epsilon
        self.epsilon = epsilon
        self.QValueTable = defaultdict(lambda: initVals)
        self.stateList = []
        self.actionList = []
        self.rewardList = []
        self.statusList = []

    def learn(self):
        nextStateQList = [value for key, value in self.QValueTable.items() if key[0] == self.nextState]
        if len(nextStateQList) == 0:
            maxQ = 0
        else:
            maxQ = max(nextStateQList)

        update = self.learningRate * (self.rewardList[-1] + self.discountFactor * maxQ - self.QValueTable[
            (self.curState, self.actionList[-1])])
        self.QValueTable[(self.curState, self.actionList[-1])] += update

        return update

    def act(self):
        randomNum = random.random()
        if randomNum < self.epsilon:
            return random.choice(self.possibleActions)
        else:
            actionDict = {key[1]: value for key, value in self.QValueTable.items() if key[0] == self.curState}
            if len(actionDict) == 0:
                return random.choice(self.possibleActions)
            else:
                return random.choice(
                    [action for action, value in actionDict.items() if value == max(actionDict.values())])

    def toStateRepresentation(self, state):
        return str(state)

    def setState(self, state):
        self.curState = state

    def setExperience(self, state, action, reward, status, nextState):
        self.stateList.append(state)
        self.actionList.append(action)
        self.rewardList.append(reward)
        self.statusList.append(status)
        self.nextState = nextState

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        self.stateList = []
        self.actionList = []
        self.rewardList = []
        self.statusList = []

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        learningRate = self.initLearningRate * 0.95 ** (episodeNumber // 100)
        epsilon = self.initEpsilon * 0.85 ** (episodeNumber // 100)

        return learningRate, epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args = parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id)
    hfoEnv.connectToServer()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1.0)
    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0
    for episode in range(numEpisodes):
        agent.reset()
        status = 0
        observation = hfoEnv.reset()

        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation
