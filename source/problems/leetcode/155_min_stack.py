# https://leetcode.com/problems/min-stack/

class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.min = []
        

    def push(self, val: int) -> None:
        self.data.append(val)
        if not self.min or val <= self.getMin():
            self.min.append(val)

    def pop(self) -> None:
        val = self.data.pop()
        if val == self.getMin():
            self.min.pop()

    def top(self) -> int:
        return self.data[-1]

    def getMin(self) -> int:
        return self.min[-1]