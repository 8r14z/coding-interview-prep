# https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        output = []
        
        def find_k_leaf_node(root, k):
            if not root or k < 0:
                return

            if k == 0:
                output.append(root.val)
                return
            
            find_k_leaf_node(root.left, k-1)
            find_k_leaf_node(root.right, k-1)
                
        def find_node(node):
            if not node:
                return 0
            
            if node == target:
                find_k_leaf_node(node, k)
                return 1
            
            count = find_node(node.left)
            is_from_left = True
            if not count:
                is_from_left = False
                count = find_node(node.right)
            
            if count:
                new_distance = k - count
                if new_distance > 0:
                    if is_from_left:
                        find_k_leaf_node(node.right, new_distance-1)
                    else:
                        find_k_leaf_node(node.left, new_distance-1)
                elif new_distance == 0:
                    output.append(node.val)

                return count + 1
                    
            return 0
                
                
        find_node(root)
        return output