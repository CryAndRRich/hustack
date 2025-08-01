from typing import List, Optional

# Definition for a Node.
class Node(object):
    def __init__(self, val: Optional[int] = None, children: Optional[List['Node']] = None):
        if children is None:
            children = []
        self.val = val
        self.children = children

class Codec:

    def serialize(self, root: 'Node') -> str:
        res = []
        self._serialize_helper(root, res)
        return ' '.join(res)

    def _serialize_helper(self, root, res):
        if not root:
            return
        res.append(str(root.val))
        res.append(str(len(root.children)))
        for child in root.children:
            self._serialize_helper(child, res)

    def deserialize(self, data: str) -> 'Node':
        if not data:
            return None
        tokens = data.split()
        self.index = 0
        return self._deserialize_helper(tokens)

    def _deserialize_helper(self, tokens):
        val = int(tokens[self.index])
        self.index += 1
        num_children = int(tokens[self.index])
        self.index += 1
        node = Node(val, [])
        for _ in range(num_children):
            node.children.append(self._deserialize_helper(tokens))
        return node

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))