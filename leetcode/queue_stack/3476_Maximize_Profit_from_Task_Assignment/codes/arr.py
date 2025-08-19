from typing import List
from collections import defaultdict

class Solution:
    def maxProfit(self, workers: List[int], tasks: List[List[int]]) -> int:
        skill_to_profits = defaultdict(list)
        for s, p in tasks:
            skill_to_profits[s].append(p)
        for s in skill_to_profits:
            skill_to_profits[s].sort(reverse=True)
        used = set()
        total = 0
        for w in workers:
            if w in skill_to_profits and skill_to_profits[w]:
                total += skill_to_profits[w][0]
                used.add((w, 0))
                skill_to_profits[w].pop(0)
        max_extra = 0
        for s in skill_to_profits:
            if skill_to_profits[s]:
                max_extra = max(max_extra, skill_to_profits[s][0])
        total += max_extra
        return total
