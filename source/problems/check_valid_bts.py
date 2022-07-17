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
            while len(stack) > 0 and a[i] > stack[-1]:
                min_val = stack.pop() # backtrack to the parent node of right-subtree
            stack.append(a[i])
            # In BST left node always has min value. So the min value is always on top of stack
            # As pre-order, we prioritize left traversal, so we keep pushing value to stack. 
            # Whenever the current node's value > the top of stack's value, it means we just turn right
            # That's when we need to track min_val as the min_value of a right-subtree should be the value of curent node, 
            # To make BST valid, the value of nodes after that should always > min_val. Otherwise it's invalid

    return 'YES'

print(isValidBTS([1, 3, 4, 2]))