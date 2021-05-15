# https://leetcode.com/problems/minimum-cost-to-move-chips-to-the-same-position/

def minCostToMoveChips(position):
    evenCount = 0
    oddCount = 0
    # move all even positions to position 2 take 0
    # move all odd positions to position 1 take 0

    for i in position:
        if i % 2 == 0:
            evenCount += 1
        else:
            oddCount += 1

    return min(evenCount, oddCount)


print(minCostToMoveChips([2,2,2,3,3]))