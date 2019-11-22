class decision_maker():

    def __init__(self, network):
        self.network = network
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None
        self.current_state = None
        self.load_and_train = True
