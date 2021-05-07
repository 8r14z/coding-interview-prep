
# https://leetfree.com/problems/wiggle-sort

# [3, 5, 2, 1, 6, 4]
# a[0] <= a[1] >= a[2] <= a[3]...

# O(nlogn)
def wiggle_sort(a):
    a.sort()
    i = 1
    while i < len(a) - 1:
        a[i], a[i+1] = a[i+1], a[i]
        i += 2
    return a

# O(n)
def wiggle_sort_improve(a):
    isMinOrder = True

    for i in range(len(a) - 1):
        if isMinOrder and a[i] > a[i+1]:
            a[i], a[i+1] = a[i+1], a[i]
        elif not isMinOrder and a[i] < a[i+1]:
            a[i], a[i+1] = a[i+1], a[i]
        isMinOrder = not isMinOrder

    return a
    
print(wiggle_sort_improve([3, 5, 2, 1, 6, 4]))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    