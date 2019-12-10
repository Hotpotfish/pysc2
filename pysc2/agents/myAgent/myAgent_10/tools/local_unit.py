class soldier():
    def __init__(self):
        self.unit_type = None
        self.alliance = None
        self.health = None
        self.energy = None
        self.x = None
        self.y = None
        self.order_length = None
        self.frend_health = []  # 附近5个友方单位的生命值
        self.enemy_health = []  # 附近5个敌方单位的生命值

    def get_list(self):
        data = [self.unit_type,
                self.alliance,
                self.health,
                self.energy,
                self.x,
                self.y,
                self.order_length] + self.frend_health + self.enemy_health
        return data
