#   1 
#     3
#      4
#     2
# invalid BTS 2 < 3
# input is an array that presents BTS in pre-order
def isValidBTS(a):
    min_val = -1
    stack = []

    for i in range(len(a)):
        if a[i] < min_val:
            return 'NO'
        else:
            while len(stack) > 0 and stack[-1] < a[i]:
                min_val = stack.pop()
            stack.append(a[i])
            # In BST left node always has min value. So the min value is always on top of stack
            # as we travel, if the top of stack has value < the current, it means we travel right
            # that's when we need to track min_val, so value of node after that should always > min_val. Otherwise it's invalid

    return 'YES'

print(isValidBTS([1, 3, 4, 2]))