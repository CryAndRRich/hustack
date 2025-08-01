from collections import defaultdict
import heapq

class TrieNode:
    def __init__(self):
        self.children = {}
        self.counts = defaultdict(int)
        self.is_end = False

class AutocompleteSystem:

    def __init__(self, sentences: list[str], times: list[int]):
        self.root = TrieNode()
        self.curr_node = self.root
        self.curr_input = ''
        for s, t in zip(sentences, times):
            self.insert(s, t)

    def insert(self, sentence: str, time: int):
        node = self.root
        for c in sentence:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.counts[sentence] += time
        node.is_end = True

    def input(self, c: str) -> list[str]:
        if c == '#':
            self.insert(self.curr_input, 1)
            self.curr_input = ''
            self.curr_node = self.root
            return []
        
        self.curr_input += c
        if self.curr_node and c in self.curr_node.children:
            self.curr_node = self.curr_node.children[c]
        else:
            self.curr_node = None
            return []

        heap = []
        for sentence, count in self.curr_node.counts.items():
            heapq.heappush(heap, (-count, sentence))
        result = []
        for _ in range(min(3, len(heap))):
            result.append(heapq.heappop(heap)[1])
        return result

# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)