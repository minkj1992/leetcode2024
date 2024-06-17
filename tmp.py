class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = [(root, float("-inf"), float("inf"))]  # Pair: node, low, high
        while stack:
            node, low, high = stack.pop()
            if not node:
                continue
            if not (low < node.val < high):
                return False
            stack.extend([(node.right, node.val, high), (node.left, low, node.val)])
        return True
