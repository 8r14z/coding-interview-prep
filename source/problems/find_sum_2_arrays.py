# arr 1 [-1,3,8,2,9,5]
# arr 2 [4,1,2,10,5,20]
# target 24
# output: (3,20) or (5,20). Distance to 24 = 1

def findMinDistance(a, k, target):
    left = 0
    right = len(a) - 1
    minDis = float('inf')

    index = -1

    while left <= right:
        mid = (left + right)//2
        sum = a[mid] + k

        dis = abs(sum-target)
        if dis < minDis:
            minDis = dis
            index = mid
    
        if sum > target:
            right = mid - 1
        elif sum < target:
            left = mid + 1
        else:
            return mid

    return [index, dis]


def findSum2Array(a1, a2, target):

    if len(a1) < 1 or len(a2) < 1: return None
    
    minDis = float('inf')
    pair = (-1,-1)
    a2.sort()

    for i, value in enumerate(a1):
        res = findMinDistance(a2, value, target)
        if res[1] < minDis:
            minDis = res[1]
            pair = (a1[i], a2[res[0]])

    return pair

print(findSum2Array([-1,3,8,2,9,5], [4,1,2,10,5,20], 24))
