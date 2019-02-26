from MDP import MDP

class BellmanDPSolver(object):
    def __init__(self):
        self.MDP = MDP()
        self.initVs()

    def initVs(self):
        self.stateValueTable = {state : 0 for state in self.MDP.S}
        self.statePolicyTale = {state : self.MDP.A for state in self.MDP.S}
        
    def BellmanUpdate(self, discount_rate):
        for state in self.MDP.S:
            action_dict = {action : sum([prob * (self.MDP.getRewards(state, action, nextState) + 
                 discount_rate * self.stateValueTable[nextState]) for nextState, prob in 
                self.MDP.probNextStates(state, action).items()]) for action in self.MDP.A}
           
            self.stateValueTable[state] = max(action_dict.values())
            self.statePolicyTale[state] = [action for action, value in action_dict.items() 
                if value == self.stateValueTable[state]]
        return self.stateValueTable, self.statePolicyTale
            
		
if __name__ == '__main__':
	solution = BellmanDPSolver()
	for i in range(20000):
		values, policy = solution.BellmanUpdate(0.9)
	print("Values : ", values)
	print("Policy : ", policy)

