import unittest

from QLearningBase import QLearningAgent


class RLTest(unittest.TestCase):
    def test_Qlearning(self):
        agent = QLearningAgent(learningRate=0.1, discountFactor=1, epsilon=1.0)
        agent.setEpsilon(1.0)
        agent.setLearningRate(0.1)

        status = 0
        for obsCopy, action, reward, nextObservation, trueUpdate in zip(
                [((1, 1), (2, 1)), ((2, 1), (2, 1)), ((1, 1), (2, 1))], ['DRIBBLE_RIGHT', 'DRIBBLE_LEFT', 'KICK'],
                [-0.4, 0.0, 1.0], [((2, 1), (2, 1)), ((1, 1), (2, 1)), 'GOAL'], [-0.04, 0.0, 0.1]):
            agent.setState(agent.toStateRepresentation(obsCopy))

            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            update = agent.learn()
            self.assertAlmostEqual(update, trueUpdate, places=7)


if __name__ == '__main__':
    unittest.main()
