# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        
        queue = deque()
        level = {root: 0}
        queue.append(root)
        
        answer = []
        tmp = []
        
        is_right_to_left = False
        
        while queue:
            node = queue.popleft()
            tmp.append(node.val)
            if not queue or level[queue[0]] != level[node]:
                if is_right_to_left:
                    tmp.reverse()
                answer.append(tmp)
                tmp = []
                is_right_to_left = not is_right_to_left
                
            if node.left: 
                level[node.left] = level[node] + 1
                queue.append(node.left)
            
            if node.right:
                level[node.right] = level[node] + 1
                queue.append(node.right)
        
        return answer