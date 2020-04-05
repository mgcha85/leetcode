class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class Solution:
    """
    easy --> hard

    """

    def get_elements(self, root, elements=[]):
        if not root:
            return elements

        if root.val is not None:
            elements.append(root.val)

        self.get_elements(root.left, elements)
        self.get_elements(root.right, elements)
        return elements

    def increasingBST(self, root: TreeNode) -> TreeNode:
        # 897. Increasing Order Search Tree (DFS)

        if not root:
            return root

        elements = sorted(self.get_elements(root))
        ans = cur = TreeNode(elements.pop(0))

        while elements:
            cur.right = TreeNode(elements.pop(0))
            cur = cur.right
        return ans

    def maxDepth(self, root, depth=1) -> int:
        # 	104 Maximum Depth of Binary Tree (DFS)
        if not root:
            return depth - 1

        ldepth = self.maxDepth(root.left, depth + 1)
        rdepth = self.maxDepth(root.right, depth + 1)
        if ldepth > rdepth:
            return ldepth
        else:
            return rdepth

    def maxDepth2(self, root, depth=0) -> int:
        # 	559. Maximum Depth of N-ary Tree (DFS)
        import collections

        if not root:
            return 0

        if not root.children:
            return depth + 1

        for child in root.children:
            depth = self.maxDepth(child, depth + 1)

        queue, depth = collections.deque(), 0
        queue.append(root)

        while queue:
            depth += 1
            size = len(queue)

            for _ in range(size):
                node = queue.popleft()
                for child in node.children:
                    queue.append(child)

        return depth

    def get_leaf(self, root, leaves=[]):
        if not root:
            return

        if root.left is None and root.right is None:
            return leaves.append(root.val)

        self.get_leaf(root.left, leaves)
        self.get_leaf(root.right, leaves)
        return leaves

    def sameList(self, l1, l2):
        if len(l1) != len(l2):
            return False

        for a, b in zip(l1, l2):
            if a != b:
                return False
        return True

    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        # 872. Leaf-Similar Trees (DFS)
        leaves1, leaves2 = [], []
        self.get_leaf(root1, leaves1)
        self.get_leaf(root2, leaves2)

        if set(leaves1) == set(leaves2) and self.sameList(leaves1, leaves2):
            return True
        else:
            return False

    def isSameTree(self, p, q):
        # 100. Same Tree (DFS)
        if not p and not q:
            return True

        if not q or not p:
            return False

        if p.val != q.val:
            return False

        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)

    def isSymmetric(self, root: TreeNode) -> bool:
        # 101. Symmetric Tree (DFS)

        def get_elements(root, elements=[], side='left'):
            if not root:
                elements.append(None)
                return elements

            elements.append(root.val)

            if side == 'left':
                self.get_elements(root.left, elements)
                self.get_elements(root.right, elements)
            else:
                self.get_elements(root.right, elements)
                self.get_elements(root.left, elements)
            return elements

        if not root:
            return True

        lele, rele = [], []
        lele = get_elements(root.left, lele)
        rele = get_elements(root.right, rele, side='right')

        if set(lele) == set(rele) and self.sameList(lele, rele):
            return True
        return False


if __name__ == '__main__':
    s = Solution()
    root = TreeNode(2)

    root.left = TreeNode(3)
    root.right = TreeNode(3)

    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    root.left.right.left = TreeNode(8)
    root.left.right.right = TreeNode(9)

    root.right.left = TreeNode(5)
    root.right.left.left = TreeNode(8)
    root.right.left.right = TreeNode(9)

    root.right.right = TreeNode(4)

    print(s.isSymmetric(root))
