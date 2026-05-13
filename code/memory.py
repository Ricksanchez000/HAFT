# memory.py

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]



class Memory_happo:
    def __init__(self):
        self.obs   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []
        self.mask     = []   
    def clear_memory(self):
        del self.obs[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.mask[:]

class OperationMemory:
    def __init__(self):
        self.obs      = []  
        self.actions  = []
        self.logprobs = []
        self.mask     = []  

    def clear(self):
        self.obs.clear(); self.actions.clear()
        self.logprobs.clear(); self.mask.clear()
