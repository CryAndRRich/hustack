class Node:
    def __init__(self):
        self.children = {}  
        self.is_file = False
        self.content = ""

class FileSystem:

    def __init__(self):
        self.root = Node()

    def ls(self, path: str):
        node = self._traverse(path)
        if node.is_file:
            return [path.split("/")[-1]]
        return sorted(node.children.keys())

    def mkdir(self, path: str):
        self._traverse(path, create=True)

    def addContentToFile(self, filePath: str, content: str):
        node = self._traverse(filePath, create=True)
        node.is_file = True
        node.content += content

    def readContentFromFile(self, filePath: str) -> str:
        node = self._traverse(filePath)
        return node.content

    def _traverse(self, path: str, create=False):
        node = self.root
        if path == "/":
            return node
        parts = path.split("/")
        for part in parts[1:]:
            if part not in node.children:
                if create:
                    node.children[part] = Node()
                else:
                    return None
            node = node.children[part]
        return node

# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.ls(path)
# obj.mkdir(path)
# obj.addContentToFile(filePath,content)
# param_4 = obj.readContentFromFile(filePath)