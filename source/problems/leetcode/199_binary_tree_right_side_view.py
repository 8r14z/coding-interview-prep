# https://leetcode.com/problems/binary-tree-right-side-view/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
    
        queue = deque([root])
        answer = []
        while queue:
            q_size = len(queue)
            node = None
            for _ in range(q_size):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                
            if node:
                answer.append(node.val)
                
        return answer
                