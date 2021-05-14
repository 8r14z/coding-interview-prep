class Solution:
    def totalFruit(self, tree: [int]) -> int:
        res = float('-inf')
        fruitMap = {} 
        start = 0

        for end, fruit in enumerate(tree):
            if fruit in fruitMap or (fruit not in fruitMap and len(fruitMap) < 2):
                fruitMap[fruit] = end
            else:
                lastOccurence = end-1
                for key, value in fruitMap.items():
                    if value < lastOccurence:
                        lastOccurence = value
                
                fruitMap.pop(tree[lastOccurence])
                fruitMap[fruit] = end
                start = lastOccurence + 1

            res = max(res, end - start + 1)

        return res

print(Solution().totalFruit([1,0,2,3,4]))