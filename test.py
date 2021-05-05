import collections






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

