# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

LEFT = '!'
RIGHT = '@'
LEFT_RIGHT = '#'
NO_CHILD = '$'
class Codec:
    def serialize(self, root):
        res = []
        def preorder(node):
            if not node:
                return
            
            res.append(str(node.val))
            left = node.left
            right = node.right
            if not left and not right:
                res.append(NO_CHILD)
            elif not left and right:
                res.append(RIGHT)
            elif left and not right:
                res.append(LEFT)
            else:
                res.append(LEFT_RIGHT)
            
            preorder(node.left)
            preorder(node.right)
        
        preorder(root)
        return ','.join(res)

    def deserialize(self, data):
        if not data:
            return None
        
        data = data.split(',')
        data.reverse()
        
        def preorder(data):
            if not data:
                return None
            
            val = int(data.pop())
            node = TreeNode(val)
            child = data.pop()
            
            if child == LEFT:
                node.left = preorder(data)
            elif child == RIGHT:
                node.right = preorder(data)
            elif child == LEFT_RIGHT:
                node.left = preorder(data)
                node.right = preorder(data)
                
            return node
            
        return preorder(data)