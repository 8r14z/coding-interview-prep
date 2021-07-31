# https://leetcode.com/problems/dot-product-of-two-sparse-vectors/

class SparseVector:
    def __init__(self, nums: List[int]):
        self.pairs = []
        for i,num in enumerate(nums):
            if num:
                self.pairs.append((i,num))

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        answer = 0
        p1 = p2 = 0
        while p1 < len(self.pairs) and p2 < len(vec.pairs):
            if self.pairs[p1][0] == vec.pairs[p2][0]:
                answer += self.pairs[p1][1] * vec.pairs[p2][1]
                p1 += 1
                p2 += 1
                
            elif self.pairs[p1][0] < vec.pairs[p2][0]:
                p1 += 1
            else:
                p2 += 1
        
        return answer
                
