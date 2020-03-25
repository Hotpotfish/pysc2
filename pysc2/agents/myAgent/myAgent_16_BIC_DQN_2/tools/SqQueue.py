# 定义队列类
class SqQueue(object):
    def __init__(self, size):
        self.size = size  # 定义队列长度
        self.real_size = 0
        self.queue = []  # 存储队列 列表
        self.head = 0

    def __str__(self):
        # 返回对象的字符串表达式，方便查看
        return str(self.queue)

    def inQueue(self, n):
        # 入队 队列满直接覆盖
        self.head = (self.head + 1) % self.size
        if self.real_size == self.size:
            self.queue[self.head] = n
        else:
            self.real_size += 1
            self.queue.append(n)

    def empty(self):
        self.queue = []

    def delete(self, n):
        # 删除某元素
        element = self.queue[n]
        self.queue.remove(element)

    def inPut(self, n, m):
        # 插入某元素 n代表列表当前的第n位元素 m代表传入的值
        self.queue[n] = m

    def getSize(self):
        # 获取当前长度
        return len(self.queue)

    def getnumber(self, n):
        # 获取某个元素
        element = self.queue[n]
        return element

    def isEmpty(self):
        # 判断是否为空
        if len(self.queue) == 0:
            return True
        return False

    def isFull(self):
        # 判断队列是否满
        if len(self.queue) == self.size:
            return True
        return False
