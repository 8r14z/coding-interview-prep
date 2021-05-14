# https://leetcode.com/problems/fruit-into-baskets/description/
class Solution:
    def totalFruit(self, tree: [int]) -> int:
        lastIndexOfFruit1 = 0
        lastIndexOfFruit2 = 0
        fruit1 = fruit2 = -1
        count = 0
        res = float('-inf')

        for i, fruit in enumerate(tree):
            if fruit1 == -1 or fruit == fruit1:
                count+=1
                fruit1 = fruit
                lastIndexOfFruit1 = i
            elif fruit2 == -1 or fruit == fruit2:
                count+=1
                fruit2 = fruit
                lastIndexOfFruit2 = i
            else:
                count = i - min(lastIndexOfFruit1, lastIndexOfFruit2)
                lastIndexOfFruit1 = max(lastIndexOfFruit1, lastIndexOfFruit2)
                lastIndexOfFruit2 = i
                fruit1 = tree[lastIndexOfFruit1]
                fruit2 = fruit

            res = max(res, count)

        return res
