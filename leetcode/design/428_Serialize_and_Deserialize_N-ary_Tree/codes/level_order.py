from typing import List, Optional
import collections

# Definition for a Node.
class Node(object):
    def __init__(self, val: Optional[int] = None, children: Optional[List['Node']] = None):
        if children is None:
            children = []
        self.val = val
        self.children = children
        
class Codec:
    def serialize(self, root: 'Node') -> str:
        if not root:
            return ""

        serialized = []
        queue = collections.deque([root, None])

        while queue:
            node = queue.popleft()
            if node is None:
                serialized.append('#')
                if queue:
                    queue.append(None)
            elif node == 'C':
                serialized.append('$')
            else:
                serialized.append(chr(node.val + 48))
                for child in node.children:
                    queue.append(child)
                if queue[0] is not None:
                    queue.append('C') 

        return ''.join(serialized)

    def deserialize(self, data: str) -> 'Node':
        if not data:
            return None

        root = Node(ord(data[0]) - 48, [])
        prev_level = collections.deque()
        curr_level = collections.deque([root])
        parent = root

        for ch in data[1:]:
            if ch == '#':
                prev_level = curr_level
                curr_level = collections.deque()
                parent = prev_level.popleft() if prev_level else None
            elif ch == '$':
                parent = prev_level.popleft() if prev_level else None
            else:
                node = Node(ord(ch) - 48, [])
                parent.children.append(node)
                curr_level.append(node)

        return root

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))