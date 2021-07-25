# https://leetcode.com/problems/random-pick-with-weight/

class Solution:

    def __init__(self, w: List[int]):
        prefix_sums = [w[0]]
        for i in range(1, len(w)):
            prefix_sums.append(w[i] + prefix_sums[i-1])
        
        self.prefix_sums = prefix_sums
        self.total = prefix_sums[-1]

    def pickIndex(self) -> int:
        target = self.total * random.random()
        result = bisect.bisect(self.prefix_sums, target)
        return result