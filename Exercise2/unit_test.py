import unittest
from QLearningBase import QLearningAgent

class Qlearning(unittest.TestCase):
    def test_Qlearning(self):
        agent = QLearningAgent(learningRate=0.1, discountFactor=1, epsilon=1.0)
        agent.setEpsilon(1.0)
        agent.setLearningRate(0.1)

        obsCopy = ((1,1),(2,1))
        agent.setState(agent.toStateRepresentation(obsCopy))
        action = 'DRIBBLE_RIGHT'
        reward = -0.4
        status = 0
        nextObservation = ((2,1),(2,1))
        agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                            agent.toStateRepresentation(nextObservation))
        update = agent.learn()
        self.assertAlmostEqual(update, -0.04, places=7)


if __name__ == '__main__':
    unittest.main()
