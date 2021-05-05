# arr 1 [-1,3,8,2,9,5]
# arr 2 [4,1,2,10,5,20]
# target 24
# output: (3,20) or (5,20). Distance to 24 = 1

def findClosestSumBetweenArrayAndKey(a, k, target):
    left = 0
    right = len(a) - 1
    minDiff = -1
    index = -1

    while left <= right:
        mid = (left + right)//2
        diff = abs(a[mid] + k - target)

        if diff > target:
            right = mid - 1
        elif diff < target:
            left = mid + 1
        else:
            return mid

        if diff < minDiff:
            minDiff = diff
            index = mid

    return index

def findClosetSumBetweenTwoArrays(array1, array2, target):
    if len(array1) < 1 or len(array2) < 1: return None
    
    minDiff = float('inf')
    pair = (-1,-1)
    array2.sort()

    for i, key in enumerate(array1):
        j = findClosestSumBetweenArrayAndKey(array2, key, target)
        curDiff = abs(array1[i] + array2[j] - target)
        if curDiff < minDiff:
            minDiff = curDiff
            pair = (array1[i], array2[j])

    return pair

print(findClosetSumBetweenTwoArrays([-1, 3, 8, 2, 9, 5], [4, 1, 2, 10, 5, 20], 24)) 
# should return (5, 20) or (3, 20).

print(findClosetSumBetweenTwoArrays([7, 4, 1, 10], [4, 5, 8, 7], 13)) 
# should return (4,8), (7, 7), (7, 5), or (10, 4).

print(findClosetSumBetweenTwoArrays([6, 8, -1, -8, -3], [4, -6, 2, 9, -3], 3)) 
# should return (-1, 4) or (6, -3).

print(findClosetSumBetweenTwoArrays([19, 14, 6, 11, -16, 14, -16, -9, 16, 13], [13, 9, -15, -2, -18, 16, 17, 2, -11, -7], -15)) 
# should return (-16, 2) or (-9, -7).
