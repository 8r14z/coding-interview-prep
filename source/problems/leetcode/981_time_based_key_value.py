# https://leetcode.com/problems/time-based-key-value-store/

class TimeMap:

    def __init__(self):
        self.storage = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.storage:
            self.storage[key] = []

        self.storage[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.storage: 
            return ""
        
        value = self.__searchValue__(self.storage[key], timestamp)
        return value if value else ""
        

    def __searchValue__(self, values, timestamp):
        if not values:
            return None
        
        left = 0
        right = len(values) - 1
        index = -float('inf')
        while left <= right:
            mid = (left + right) // 2
            midTimestamp = values[mid][0]
            if midTimestamp == timestamp:
                return values[mid][1]
            elif midTimestamp > timestamp:
                right = mid - 1
            elif midTimestamp < timestamp:
                index = max(index, mid)
                left = mid + 1

        return values[index][1] if index > -1 else None

