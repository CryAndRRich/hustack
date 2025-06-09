## Description
You are given **N** cows on a straight line at positions $x_1, x_2, ..., x_N$ $(0 \leq x_i \leq 1,000,000,000)$. You must choose **C** cows $(2 \leq C \leq N)$ such that the **minimum distance** between any two chosen cows is **maximized**

**Input:**
- The first line contains an integer $T$ $(1 \leq T \leq 20)$, the number of test cases.
- For each test case:
    - Line 1: Two space-separated integers $N$ and $C$ — the number of cows and the number of cows to select
    - Line 2: $N$ space-separated integers $x_1, x_2, ..., x_N$ — the positions of the cows

**Output:** For each test case, print a single integer — the largest minimum distance possible between any two of the selected $C$ cows

## Approach: Binary Search + Greedy
We can perform **binary search** on the answer (i.e., the minimum distance $d$) and use a **greedy check** to determine if it's possible to place $C$ cows with at least that distance
- Sort the array of positions
- Binary search over $d \in [1, max(x) - min(x)]$
- For each mid $d$, greedily place cows: start from the first position and always place the next cow at the first valid position ≥ last placed + d
- If we can place $C$ cows, try a larger distance. Otherwise, try smaller  

**Time Complexity:**  
- Sorting: $O(N log N)$
- Binary Search: $O(log(max\_distance))$
- Greedy check per step: $O(N)$
- Overall per test case: **$O(N \times log(max\_distance))$**  

**Space Complexity:** $O(N)$ for storing positions
