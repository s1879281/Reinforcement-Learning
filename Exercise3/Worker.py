import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random

def train():
    pass



def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    if done:
        return reward
    else:
        target_value = reward + discountFactor * max(targetNetwork(nextObservation)[0])
        return target_value


def computePrediction(state, action, valueNetwork):
    pred_value = valueNetwork(state)[0][action]
    return pred_value

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)





