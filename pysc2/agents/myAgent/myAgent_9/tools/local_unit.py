class soldier():
    def __init__(self):
        self.unit_type = None
        self.alliance = None
        self.health = None
        self.shield = None
        self.energy = None
        self.x = None
        self.y = None
        self.order_length =None
        # self.frend = []  # 附近5个友方实体
        # self.near = []  # 附近5个友方实体

    def get_list(self):
        temp = [self.unit_type,
                self.alliance,
                self.health,
                self.shield,
                self.energy,
                self.x,
                self.y,
                self.order_length]
        return temp
