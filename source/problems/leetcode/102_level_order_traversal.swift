// https://leetcode.com/problems/binary-tree-level-order-traversal/
class Solution {
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else {
            return []
        }

        var queue: [TreeNode] = [root]
        var result: [[Int]] = []

        while queue.count > 0 {
            var nextLevelQueue: [TreeNode] = []
            var currentLevelResult: [Int] = []
            for node in queue {
                currentLevelResult.append(node.val)

                if let leftChild = node.left {
                    nextLevelQueue.append(leftChild)
                }
                if let rightChild = node.right {
                    nextLevelQueue.append(rightChild)
                }
            }
            result.append(currentLevelResult)
            queue = nextLevelQueue
        }

        return result
    }
}