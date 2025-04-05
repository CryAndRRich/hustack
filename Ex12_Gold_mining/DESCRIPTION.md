## Problem Description
The Kingdom ALPHA has **n** warehouses of gold located on a straight line and numbered $1, 2, \dots, n$. Each warehouse $i$ contains an amount of gold denoted by $a_i$ (where $a_i$ is a non-negative integer) and is located at coordinate $i$. The King of ALPHA holds a competition for hunters to select a subset of these warehouses such that the **total amount of gold** is maximized. However, there is a constraint: the distance between any two selected warehouses must be **greater than or equal to $L_1$** and **less than or equal to $L_2$**

**Input:**
- **Line 1:** Three integers $n$, $L_1$, and $L_2$ ($1 \leq n \leq 1000000$, $1 \leq L_1 \leq L_2 \leq n$)
- **Line 2:** $n$ integers representing the amounts of gold: $a_1, a_2, \dots, a_n$

**Output:**
- A single integer representing the maximum total amount of gold that can be obtained by selecting a subset of warehouses under the given distance condition

## Approach 1: Backtracking + Branch and Bound (Failed)
This approach explores all possible subsets of warehouses that satisfy the distance constraints and computes the sum of gold for each valid subset
- **Recursive Exploration:** The algorithm recursively decides whether to include a warehouse. When including warehouse $i$, the next possible warehouse to be included must have an index $j$ such that $j - i \geq L_1$ and $j - i \leq L_2$
- **Validation:** At each step, the algorithm ensures that the selected warehouses satisfy the distance condition
- **Result:** It keeps track of the maximum total amount of gold obtained from any valid subset

**Time Complexity:**
- Worst-case exponential due to exploring all possible subsets, approximately $O(2^n)$ in the worst case 
- Pruning based on the distance constraint can reduce the practical search space

**Space Complexity:**
- $O(n)$ for the recursion stack and storing the current subset

## Approach 2: Dynamic Programming
Dynamic Programming (DP) can be used to optimize the solution by breaking the problem into subproblems. We explore several DP-based methods to handle the constraint on the distance between selected warehouses

### Ver1: Pure DP
Let $dp[i]$ be the maximum total gold that can be collected ending at warehouse $i$. For each warehouse $i$, consider all possible previous warehouses $j$ such that $i - j \geq L_1$ and $i - j \leq L_2$. Then, $dp[i] = \max(dp[i], dp[j] + a_i)$. The final answer is the maximum value in the $dp$ array

- **Time Complexity:** $O(n \times (L_2 - L_1 + 1))$, which may be inefficient for large $n$
- **Space Complexity:** $O(n)$ for the DP array

### Ver 2: DP + Queue
Use a deque (queue) to maintain the maximum $dp[j]$ for indices $j$ in the sliding window corresponding to the valid range [$i-L_2$, $i-L_1$]. For each warehouse $i$, the deque helps to quickly get the maximum $dp[j]$ in the allowed window and update $dp[i] = a_i + \text{(maximum from the queue)}$. This yields the optimal value in an efficient manner

- **Time Complexity:** $O(n)$, as each element is added and removed from the deque at most once
- **Space Complexity:** $O(n)$ for the DP array and the deque

### Ver3: DP + Segment Tree
Use a segment tree to perform range maximum queries. For each warehouse $i$, query the segment tree for the maximum $dp[j]$ in the range corresponding to indices $[i - L_2, i - L_1]$. Update $dp[i] = a_i + \text{(result from segment tree)}$. After calculating $dp[i]$, update the segment tree at index $i$ with the new value. The maximum total gold can be obtained from the DP array after processing all warehouses

- **Time Complexity:** Each range query and update operation takes $O(\log n)$, so overall time is $O(n \log n)$
- **Space Complexity:** $O(n)$ for the DP array and $O(n)$ for the segment tree

### Ver4: DP + Binary Indexed Tree
Use a Fenwick Tree (Binary Indexed Tree) to perform efficient range maximum queries. For each warehouse $i$, use the Fenwick Tree to query the maximum $dp[j]$ for indices in the range $[i - L_2, i - L_1]$. Then, set $dp[i] = a_i + \text{(queried maximum)}$. Update the Fenwick Tree at position $i$ with $dp[i]$. The final answer is the maximum value in the DP array

- **Time Complexity:** Each query and update is $O(\log n)$, leading to an overall time of $O(n \log n)$
- **Space Complexity:** $O(n)$ for both the DP array and the Fenwick Tree

## Result:
<table style="text-align: center;">
    <thead>
        <tr>
            <th style="text-align: center;">File</th>
            <th colspan="2" style="text-align: center;">Test 1</th>
            <th colspan="2" style="text-align: center;">Test 2</th>
            <th colspan="2" style="text-align: center;">Test 3</th>
            <th colspan="2" style="text-align: center;">Test 4</th>
            <th colspan="2" style="text-align: center;">Test 5</th>
            <th colspan="2" style="text-align: center;">Test 6</th>
        </tr>
        <tr>
            <th></th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>backtrack.c</td>
            <td>3779</td>
            <td><strong>1.391</strong></td>
            <td>6</td>
            <td>0.77</td>
            <td>32</td>
            <td>0.793</td>
            <td>382</td>
            <td>1.48</td>
            <td>5</td>
            <td>0.77</td>
            <td>TLE</td>
            <td></td>
        </tr>
        <tr>
            <td>dp(ver1).py</td>
            <td>TLE</td>
            <td></td>
            <td>59</td>
            <td>3.234</td>
            <td>986</td>
            <td>3.629</td>
            <td>TLE</td>
            <td></td>
            <td>41</td>
            <td>3.191</td>
            <td>TLE</td>
            <td></td>
        </tr>
        <tr>
            <td>dp(ver2).py</td>
            <td>262</td>
            <td>8.676</td>
            <td>47</td>
            <td>3.473</td>
            <td>71</td>
            <td>3.984</td>
            <td>252</td>
            <td>8.746</td>
            <td>45</td>
            <td>3.508</td>
            <td>TLE</td>
            <td></td>
        </tr>
        <tr>
            <td>dp(ver2).c</td>
            <td>29</td>
            <td>1.469</td>
            <td><strong>4</strong></td>
            <td><strong>0.719</strong></td>
            <td><strong>8</strong></td>
            <td><strong>0.766</strong></td>
            <td><strong>22</strong></td>
            <td><strong>1.469</strong></td>
            <td>5</td>
            <td>0.844</td>
            <td>128</td>
            <td>8.43</td>
        </tr>
        <tr>
            <td>dp(ver2).cpp</td>
            <td><strong>25</strong></td>
            <td>1.582</td>
            <td>6</td>
            <td>0.969</td>
            <td>9</td>
            <td>0.891</td>
            <td>35</td>
            <td>1.574</td>
            <td>8</td>
            <td>0.828</td>
            <td><strong>103</strong></td>
            <td><strong>8.406</strong></td>
        </tr>
        <tr>
            <td>dp(ver3).py</td>
            <td>TLE</td>
            <td></td>
            <td>58</td>
            <td>3.246</td>
            <td>216</td>
            <td>4.059</td>
            <td>TLE</td>
            <td></td>
            <td>38</td>
            <td>3.273</td>
            <td>TLE</td>
            <td></td>
        </tr>
        <tr>
            <td>dp(ver3).c</td>
            <td>50</td>
            <td>2.418</td>
            <td>6</td>
            <td>0.801</td>
            <td>10</td>
            <td>1.027</td>
            <td>49</td>
            <td>2.391</td>
            <td><strong>4</strong></td>
            <td><strong>0.723</strong></td>
            <td>293</td>
            <td>16.355</td>
        </tr>
        <tr>
            <td>dp(ver3).cpp</td>
            <td><strong>25</strong></td>
            <td>2.699</td>
            <td>7</td>
            <td>0.828</td>
            <td>11</td>
            <td>1.039</td>
            <td>24</td>
            <td>2.59</td>
            <td>7</td>
            <td>0.84</td>
            <td>218</td>
            <td>16.465</td>
        </tr>
        <tr>
            <td>dp(ver4).c</td>
            <td>55</td>
            <td>1.867</td>
            <td>6</td>
            <td>0.813</td>
            <td>10</td>
            <td>0.844</td>
            <td>61</td>
            <td>1.938</td>
            <td>5</td>
            <td>0.965</td>
            <td>436</td>
            <td>12.199</td>
        </tr>
        <tr>
            <td>dp(ver4).cpp</td>
            <td>42</td>
            <td>1.984</td>
            <td>8</td>
            <td>0.84</td>
            <td>11</td>
            <td>0.961</td>
            <td>18</td>
            <td>1.941</td>
            <td>8</td>
            <td>0.809</td>
            <td>252</td>
            <td>12.195</td>
        </tr>
    </tbody>
</table>
