import collections
from source.algorithms.quick_sort import partition

def greatestCommonDivisor(a, b):
    if a == 0:
        return b
    if b == 0:
        return a

    remainder = a % b
    return greatestCommonDivisor(b, remainder)


def longest_common_subsequence(str1, str2):
    if not str1 or not str2:
        return 0

    newStr1 = str1[0:len(str1) - 1]
    newStr2 = str2[0:len(str2) - 1]

    if str1[-1] == str2[-1]:
        return 1 + longest_common_subsequence(newStr1, newStr2)
    else:
        return max(longest_common_subsequence(newStr1, str2), longest_common_subsequence(str1, newStr2))

# print(longest_common_subsequence('abdjsh', 'ads')) # a d s


def longest_common_subsequence_dp(str1, str2):
    res = 0
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
            res = max(res, dp[i][j])

    return res

# print(longest_common_subsequence_dp('fort', 'fosh')) # a d s


def longest_common_substring_dp(str1, str2):
    res = 0
    dp = [[0] * len(str2) for _ in range(len(str1))]

    for i in range(len(str1)):
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                dp[i][j] = 1 + (dp[i-1][j-1] if i > 0 and j > 0 else 0)
            else:
                dp[i][j] = 0
            res = max(res, dp[i][j])
    print(dp)
    return res

# print(longest_common_substring('fish', 'hish'))
# print(longest_common_substring('fisdishe', 'hishefcd'))


def findKthLargest(arr, k):
    left = 0
    right = len(arr) - 1

    while left <= right:
        pivotIndex = partition(arr, left, right)

        if pivotIndex == len(arr) - k:
            return pivotIndex
        elif pivotIndex > len(arr) - k:
            right = pivotIndex - 1
        else:
            left = pivotIndex + 1
    return -1

# print(findKthLargest([5,7,2,3,4, 1,6], 3))


def merge2sorted(arr1, arr2):
    newArr = []
    n1 = len(arr1)
    n2 = len(arr2)
    i = 0
    j = 0

    while i + j < n1 + n2:
        if i < n1 and j < n2:
            if arr1[i] <= arr2[j]:
                newArr.append(arr1[i])
                i += 1
            else:
                newArr.append(arr2[j])
                j += 1
        elif i >= n1:
            newArr.append(arr2[j])
            j += 1
        else:
            newArr.append(arr1[i])
            i += 1

    return newArr

# print(merge2sorted([1,4,5, 7], [2,3,4,6]))


def max_sum_subarray(arr, k):
    max_sum = -1
    start = 0
    cur_sum = 0

    for end, val in enumerate(arr):
        cur_sum += val
        if end - start + 1 == k:
            max_sum = max(max_sum, cur_sum)
            cur_sum -= arr[start]
            start += 1

    return max_sum

# print(max_sum_subarray([2, 3, 4, 1, 5], 3))


def smallest_size_subarray(arr, k):
    min_size = -1
    start = 0
    cur_sum = 0
    for end, val in enumerate(arr):
        cur_sum += val
        while cur_sum >= k:
            min_size = end - start + 1 if min_size == - \
                1 else min(min_size, end - start + 1)
            cur_sum -= arr[start]
            start += 1

    return min_size

# print(smallest_size_subarray([2, 4, 2, 5, 1], 7))


def find_max_price(price_arr, weight_arr, max_weight):
    max_price = float('-inf')
    start = 0
    cur_weight = 0
    cur_price = 0

    for end in range(len(price_arr)):
        cur_weight += weight_arr[end]
        cur_price += price_arr[end]
        while cur_weight > max_weight:
            cur_weight -= weight_arr[start]
            cur_price -= price_arr[start]
            start += 1

        max_price = max(cur_price, max_price)

    return max_price

# print(find_max_price([3000, 3000, 2000, 1500], [36, 30, 20, 15], 35))


def longestPalindromeDp(s):
    rev = s[::-1]
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    max_len = 0
    max_end = 0

    for i in range(n):
        for j in range(n):
            if s[i] == rev[j]:
                dp[i][j] = 1 + (dp[i-1][j-1] if i > 0 and j > 0 else 0)
                cur += s[i]
            else:
                dp[i][j] = 0
            if dp[i][j] > max_len and i - dp[i][j] + 1 == len(s) - 1 - j:
                # start index of duplicated substring [x] should be same as index  on reversed string in revered order
                # index x = 2 -> y = len(s) - 1 - x
                max_len = dp[i][j]
                max_end = i

    return s[max_end - max_len + 1: max_end+1]

# print(longestPalindrome("babad"))


def removeInvalidParentheses(s):
    invalid_indices = set()
    parentheses = []

    for i, c in enumerate(s):
        if c == '(':
            parentheses.append(i)
        elif c == ')':
            if len(parentheses) == 0:
                invalid_indices.add(i)
            else:
                parentheses.pop()

    res = ''
    for i in range(len(s)):
        if i not in parentheses and i not in invalid_indices:
            res += s[i]

    return res

# print(removeInvalidParentheses("lee(t(c)o)de)"))


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

    return 'YES'

# print(isValidBTS([1, 3, 4, 2]))
# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 != None and l2 is None:
            return l1
        elif l2 != None and l1 is None:
            return l2
        elif l1 is None and l2 is None:
            return None
        
        head = None
        if l1.val < l2.val: 
            head = l1
            l1 = l1.next
        else:
            head = l2
            l2 = l2.next
        head.next = self.mergeTwoLists(l1, l2)

        return head

    def longestCommonPrefix(self, strs: [str]) -> str:
        res = ''
        
        for i in range(1, len(strs)):
            prefix = self.commonPrefix(strs[i], strs[i-1])
            if res == '' or len(prefix) < len(res):
                res = prefix
        
        return res
    
    def commonPrefix(self, str1, str2):
        n = min(len(str1), len(str2))
        prefix = ''
        for i in range(n):
            if str1[i] != str2[i]:
                break
            prefix += str1[i]
        return prefix

print(Solution().longestCommonPrefix(['dog', 'racecard', 'card']))


# Heap sort

