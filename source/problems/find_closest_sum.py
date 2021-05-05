# arr 1 [-1,3,8,2,9,5]
# arr 2 [4,1,2,10,5,20]
# target 24
# output: (3,20) or (5,20). Distance to 24 = 1

def findClosestSumBetweenArrayAndKey(a, k, target):
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
            return mid, minDis

    return [index, minDis]


def findClosetSumBetweenTwoArrays(array1, array2, target):

    if len(array1) < 1 or len(array2) < 1: return None
    
    minDis = float('inf')
    pair = (-1,-1)
    array2.sort()

    for i, key in enumerate(array1):
        res = findClosestSumBetweenArrayAndKey(array2, key, target)
        if res[1] < minDis:
            minDis = res[1]
            pair = (array1[i], array2[res[0]])

    return pair

print(findClosetSumBetweenTwoArrays([-1,3,8,2,9,5], [4,1,2,10,5,20], 24))
