# https://www.educative.io/blog/crack-amazon-coding-interview-questions

import math
from collections import deque
from functools import lru_cache

# find the missing number in sequence from 1 to n
def find_missing(input):
    n = len(input)+1
    if n < 1:
        return -1

    actual_sum = sum(input)
    expected_sum = (n * (n+1))//2

    return expected_sum - actual_sum

# two sum
def find_sum_of_two(A, val):
    n = len(A)
    if n < 2:
        return False
    seen = set()
    for num in A:
        if (val - num) in seen:
            return True
        seen.add(num)
    return False

# merge 2 sorted lists
def merge_sorted(head1, head2):
    if head1 == None and head2 == None:
        return None
    elif head1 == None:
        return head2
    elif head2 == None:
        return head1

    new_head = None
    if head1.data < head2.data:
        new_head = head1
        new_head.next = merge_sorted(head1.next, head2)
    else:
        new_head = head2
        new_head.next = merge_sorted(head1, head2.next)

    return new_head

# deep copy a linked list with arbitrary pointer
class LinkedListNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.arbitrary = None

def deep_copy_arbitrary_pointer(head):
    # LinkedListNode data - next
    new_head = LinkedListNode(head.data)
    old_head = head

    new_running_pointer = new_head
    links = {old_head: new_head}

    while old_head.next:
        next = old_head.next
        new_node = LinkedListNode(next.data)

        new_node.arbitrary = next.arbitrary
        new_running_pointer.next = new_node

        old_head = next
        new_running_pointer = new_node

        if old_head:
            links[old_head] = new_running_pointer

    current = new_head
    while current:
        if current.arbitrary:
            current.arbitrary = links[current.arbitrary]
        current = current.next

    return new_head

# print each level for binary tree
def level_order_traversal(root):
    result = ''
    queue = deque()
    queue.append(root)

    while queue:
        node = queue.popleft()
        result += str(node.data) + ' '

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result


INF = float('inf')

# check if tree is a BST
def is_bst(root, min=-INF, max=INF):
    # data - left - right
    if not root:
        return True

    if root.data < min or root.data > max:
        return False

    return is_bst(root.left, min, root.data) and is_bst(root.right, root.data, max)

# string segmentation
# DP, the current word in dictionary so the string is segmentable if the next word is in dictionary
def can_segment_string(s, dictionary):
    n = len(s)
    dp = [False] * n

    for i in reversed(range(n)):
        for j in range(i, n):
            word = s[i:j+1]
            if word in dictionary:
                if j == n-1 or dp[j+1]:
                    dp[i] = True
                    break
    return dp[0]

# reverse words in a sentence
def reverseWords(s: str) -> str:
    n = len(s)
    start = -1
    end = -1
    res = ''
    for i in reversed(range(n)):
        if i == n-1 or (s[i] != ' ' and s[i+1] == ' '):
            end = i

        if i == 0 or (s[i] != ' ' and s[i-1] == ' '):
            start = i

        if start != -1 and end != -1 and start <= end:
            res += (s[start: end+1] + ' ')
            start = end = -1

    return res


print(reverseWords('Hello     world'))


# play dumb and eventually solve with constant space :))
def solve_coin_change(denominations, amount):
    n = len(denominations)
    dp = [[0] * n for _ in range(amount + 1)]

    for i in range(amount+1):
        for j in range(n):
            if i == 0:
                dp[0][j] = 1
                continue

            cur_coin = denominations[j]
            sum_include_coin = dp[i-cur_coin][j] if i >= cur_coin else 0
            sum_exclude_coin = dp[i][j-1] if j >= 1 else 0

            dp[i][j] = sum_include_coin + sum_exclude_coin

    return dp[amount][n-1]

# find K-th permutation in a set of n elements
# {1,2,3}
# 123, 132, 213, 231, 312, 321
def find_kth_permutation(v, k):
    if not v:
        return ''

    n = len(v)

    if n == 1:
        return str(v[0])

    block_size = math.factorial(n-1)
    selected_index = math.ceil(k/math.factorial(n-1))-1
    res = str(v[selected_index])
    del v[selected_index]
    return res + find_kth_permutation(v, k - (selected_index*block_size))

print(find_kth_permutation([1, 2, 3], 1))
print(find_kth_permutation([1, 2, 3], 2))
print(find_kth_permutation([1, 2, 3], 3))
print(find_kth_permutation([1, 2, 3], 4))
print(find_kth_permutation([1, 2, 3], 5))
print(find_kth_permutation([1, 2, 3], 6))

# get all subsets of a given array
# generate all subsets by 2**n bitmask 
def get_all_subsets(v, sets):
    n = len(v)
    for mask in range(2**n):
        subset = set()
        for i in range(n):
            if mask & (1 << i) > 0:
                subset.add(v[i])

        sets.append(subset)

    return sets

# print all balanced braces with n opens and n closes
def print_all_braces(n):
    result = []

    def backtrack(open, close, string):
        if open == close == 0:
            result.append(string)

        if open == close:
            backtrack(open-1, close, string + '{')
        elif open == 0:
            string += '}'
            backtrack(0, close-1, string + '}')
        else:
            backtrack(open, close-1, string + '}')
            backtrack(open-1, close, string + '{')

    return backtrack(n, n, '')

# deep copy a graph
# same technique -> use a hash to map betwen the one in old graph to new graph
class Node:
    def __init__(self, d):
        self.data = d
        self.neighbors = []


def clone(root):
    new_root = Node(root.data)
    queue = deque()
    queue.append(root)
    cloned = {root: new_root}

    while queue:
        node = queue.popleft()
        for neighbor in node.neighbors:
            if neighbor not in cloned:
                new_node = Node(neighbor.data)
                queue.append(neighbor)
                cloned[neighbor] = new_node

        for neighbor in node.neighbors:
            cloned[node].neighbors.append(cloned[neighbor])

    return new_root    # return root


# given an array with duplicated integers
# find lowest index and highest index
INF = float('inf')
def _find_index(arr, key, is_low):
    n = len(arr)
    left = 0
    right = n-1
    res = INF if is_low else -INF

    while left <= right:
        mid = (left+right) // 2
        if arr[mid] > key:
            right = mid - 1
        elif arr[mid] < key:
            left = mid + 1
        else:
            if is_low:
                res = min(res, mid)
                right = mid - 1
            else:
                res = max(res, mid)
                left = mid + 1

    return res if res >= 0 and res < n else -1

def find_low_index(arr, key):
    return _find_index(arr, key, True)

def find_high_index(arr, key):
    return _find_index(arr, key, False)

# given an array that can be modified by a "number" of times
# i,g: [1,2,3,4] -> [4,1,2,3] -> modify 1 time
def binary_search_rotated(arr, key):
    def find_min_index(arr):
        left = 0
        right = len(arr) - 1
        last = arr[-1]
        res = float('inf')
        while left <= right:
            mid = (left+right) // 2
            if arr[mid] > last:
                left = mid+1
            else:
                res = min(res, mid)
                right = mid-1

        return res

    min_index = find_min_index(arr)
    left = right = -1
    if key > arr[-1]:
        left = 0
        right = min_index-1
    else:
        left = min_index
        right = len(arr) - 1

    while left <= right:
        mid = (left+right) // 2
        if arr[mid] > key:
            right = mid - 1
        elif arr[mid] < key:
            left = mid + 1
        else:
            return mid
    return -1


# find k largest elemetn
# O(n) on average - O(n**2) in worst case
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k > n: return -1
        
        def partition(left, right, arr) -> int:
            pivot = arr[right]
            pivot_index = left # save last index that > pivot
            
            for i in range(left,right):
                if arr[i] <= pivot:
                    arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
                    pivot_index += 1
                    
            arr[right], arr[pivot_index] = arr[pivot_index], arr[right]
            
            return pivot_index
            
        left = 0
        right = n-1
        target = n-k
        
        while left <= right:
            pivot_index = partition(left, right, nums)
            if pivot_index == target:
                return nums[pivot_index]
            elif pivot_index > target:
                right = pivot_index - 1
            else:
                left = pivot_index + 1
                
        return -1
        

# convert a BST to DLL
def convert_to_linked_list(root):
    def inorder(node):
        if not node: return None, None
        if not node.left and not node.right:
            return node, node

        left = node.left
        right = node.right

        l_head, l_tail = inorder(left)
        r_head, r_tail = inorder(right)

        if l_tail: 
            l_tail.right = node
        node.left = l_tail
        node.right = r_head
        if r_head: 
            r_head.left = node

        return l_head if l_head else node, r_tail if r_tail else node
    
    head,_ = inorder(root)
    return head


# Given a binary tree T, find the maximum path sum. The path may start and end at any node in the tree.
class TreeNode:
    def __init(self, val: int):
        self.val = val
        self.left = None
        self.right = None
def maxPathSum(self, root: TreeNode) -> int:
    def dfs(node) -> int:
        nonlocal global_max
        if not node: 
            return 0
            
        cur_val = node.val
        left_val = dfs(node.left)
        right_val = dfs(node.right)
            
        cur_max = max([
            cur_val, 
            cur_val + left_val, 
            cur_val + right_val, 
            cur_val + left_val + right_val
        ])
                       
        global_max = max(global_max, cur_max)
            
        return max(cur_val, cur_val + left_val, cur_val + right_val)
            # sequence means the path should be in same branch == A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
        
    global_max = float('-inf')
    dfs(root)
    return global_max


# rotate matrix by 90 degress
def rotate(matrix: [[int]]):
    n = len(matrix)
    if n < 2:
        return
    
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    for i in range(n):
        for j in range(n//2):
            matrix[i][j], matrix[i][n-j-1] = matrix[i][n-j-1], matrix[i][j]


# rotate and array by k
def rotate_arr(nums, k):
    def reverse(start, end, arr):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    n = len(nums)
    k = k%n
    if k == 0:
        return
    nums.reverse()
    reverse(0, k-1, nums)
    reverse(k, n-1, nums)

