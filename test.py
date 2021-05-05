import collections


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

