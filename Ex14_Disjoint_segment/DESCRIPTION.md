## Description
Given a set of segments $X = ((a_1, b_1), \dots, (a_n, b_n))$ in which $a_i < b_i$ are coordinates of the segment $i$ on a line, $i = 1,\dots, n$.  Find a subset of $X$ having largest cardinality in which no two segments of the subset intersect

**Input:**
- **Line $1$:** contains a positive integer $n$ $(1 \leq n \leq 100000)$
- **Line $i+1$ $(i = 1, \dots , n)$:** contains $a_i$ and $b_i$ $(0 \leq a_i \leq b_i \leq 1000000)$

**Output:** Number of segments in the solution found

## Approach 1: DP + Binary Search
Sorts the segments by their end coordinate. For each segment $i$:
- It finds the last segment (with index $ind$) that ends at or before the start of segment $i$ (using binary search)
- It then sets a value $candidate = 1+dp[ind+1]$ (if such an interval exists) or just $1$ otherwise.
- Finally, it updates $dp[i+1]$ to be the maximum of $dp[i]$ (not including segment $i$) and the candidate (including it)

**Time Complexity:** $O(nlogn)$

**Space Complexity:** $O(n)$

## Approach 2: Greedy Algorithm
Sorts the segments by their end coordinate. Iterate through the segments and select the current segment if it does not overlap with the last selected segment

**Time Complexity:** $O(n)$

**Space Complexity:** $O(n)$


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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>dp.py</td>
            <td>42</td>
            <td>3.223</td>
            <td>54</td>
            <td>3.262</td>
            <td>60</td>
            <td>3.242</td>
            <td>981</td>
            <td>23.949</td>
            <td>143</td>
            <td>5.199</td>
        </tr>
        <tr>
            <td>dp.c</td>
            <td><strong>6</strong></td>
            <td><strong>0.715</strong></td>
            <td><strong>7</strong></td>
            <td><strong>0.719</strong></td>
            <td><strong>6</strong></td>
            <td><strong>0.715</strong></td>
            <td>128</td>
            <td>5.363</td>
            <td>18</td>
            <td><strong>1.094</strong></td>
        </tr>
        <tr>
            <td>dp.cpp</td>
            <td>8</td>
            <td>0.922</td>
            <td>8</td>
            <td>0.844</td>
            <td><strong>6</strong></td>
            <td>1</td>
            <td>127</td>
            <td>6.582</td>
            <td>19</td>
            <td>1.414</td>
        </tr>
        <tr>
            <td>greedy.py</td>
            <td>42</td>
            <td>3.195</td>
            <td>50</td>
            <td>3.383</td>
            <td>46</td>
            <td>3.133</td>
            <td>559</td>
            <td>24.02</td>
            <td>104</td>
            <td>5.109</td>
        </tr>
        <tr>
            <td>greedy.c</td>
            <td><strong>6</strong></td>
            <td>0.797</td>
            <td><strong>7</strong></td>
            <td>0.98</td>
            <td><strong>6</strong></td>
            <td>0.719</td>
            <td>86</td>
            <td><strong>5.313</strong></td>
            <td><strong>15</strong></td>
            <td>1.172</td>
        </tr>
        <tr>
            <td>greedy.cpp</td>
            <td>8</td>
            <td>0.863</td>
            <td>9</td>
            <td>0.855</td>
            <td>9</td>
            <td>0.832</td>
            <td><strong>82</strong></td>
            <td>6.195</td>
            <td>16</td>
            <td>1.379</td>
        </tr>
    </tbody>
</table>
