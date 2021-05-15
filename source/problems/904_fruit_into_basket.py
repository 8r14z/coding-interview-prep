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


class Solution:
    def totalFruit(self, tree: [int]) -> int:
        if tree is None or len(tree) < 1: return 0
        
        res = float('-inf')
        basket = {} 
        start = 0

        for end, fruit in enumerate(tree):
            if fruit in basket or (fruit not in basket and len(basket) < 2):
                basket[fruit] = end
            else:
                minIndex = end-1
                for key, value in basket.items():
                    if value < minIndex:
                        minIndex = value
                
                basket.pop(tree[minIndex])
                basket[fruit] = end
                start = minIndex + 1

            res = max(res, end - start + 1)

        return res
