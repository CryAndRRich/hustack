## Description
Given a sequence $a$ of positive $a_1, a_2, ..., a_n$. A subsequence of $a$ is defined to be the sequence obtained by removing some elements. Find the subsequence of $a$ such that the elements is in an increasing order and the number of elements of the subsequence is maximal

**Input:**
- **Line 1**: Contains a positive integer $n$ $(1 \leq n \leq 10^5)$
- **Line 2**: Contains $n$ elements of the given sequence $(1 \leq a_i \leq 10^5)$

**Output:** The length of the subsequence found

## Approach 1: Patience Sorting
Use the **patience sorting** method to find the length of the Longest Increasing Subsequence:
- Maintain an array $tails[]$, initially empty
- For each element $x$ in the input sequence:
    - Binary‐search $tails$ for the **lowest index** $i$ such that $tails[i] \geq x$
    - If none found (i.e. $x$ is larger than all current tails), **append** $x$ to $tails$
    - Otherwise, **replace** $tails[i]$ with $x$
- The length of $tails$ at the end is the length of the LIS

**Time Complexity:** $O(nlogn)$

**Space Complexity:** $O(n)$

## Approach 2: DP + Segment (Fenwich) Tree
- Coordinate Compression:
    - Collect all $a[i]$ into an array, sort and remove duplicates, size $m$
    - Map each $a[i]$ to its index $idx$ in the compressed array (range $[0..m-1]$)
- Define $dp[i]$ = length of the longest increasing subsequence ending at index $i$  
- Process elements in original order; for each $a[i]$ with compressed index $idx$:
    - Query segment tree for $max(dp[j])$ over all values $< a[i]$, i.e. range $[0..idx-1]$.  
    - Let $cur = (that max) + 1$ (or 1 if $idx == 0$).  
    - Update segment tree at position $idx$ with $cur$ (take the maximum if multiple updates).  
- Track the global maximum of all $cur$
- The maximum $dp[i]$ over all $i$ is the length of the LIS

**Time Complexity:** $O(nlogn)$
- Coordinate compression: $O(nlogn)$  
    - For each of n elements:
        - Segment tree query: $O(logm)$  
        - Segment tree update: $O(logm)$  

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
            <td>patience.py</td>
            <td>68</td>
            <td>4.18</td>
            <td>166</td>
            <td>15.711</td>
            <td>75</td>
            <td>5.293</td>
            <td>43</td>
            <td>4.309</td>
            <td>65</td>
            <td>4.383</td>
        </tr>
        <tr>
            <td>patience.c</td>
            <td><strong>6</strong></td>
            <td>0.887</td>
            <td>42</td>
            <td><strong>1.102</strong></td>
            <td>9</td>
            <td><strong>0.715</strong></td>
            <td><strong>4</strong></td>
            <td><strong>0.711</strong></td>
            <td>7</td>
            <td><strong>0.711</strong></td>
        </tr>
        <tr>
            <td>patience.cpp</td>
            <td>8</td>
            <td>0.895</td>
            <td><strong>21</strong></td>
            <td>1.25</td>
            <td><strong>8</strong></td>
            <td>1.09</td>
            <td>8</td>
            <td>0.797</td>
            <td>8</td>
            <td>0.898</td>
        </tr>
        <tr>
            <td>segtree.py</td>
            <td>44</td>
            <td>3.258</td>
            <td>TLE</td>
            <td></td>
            <td>221</td>
            <td>4.813</td>
            <td>44</td>
            <td>3.316</td>
            <td>63</td>
            <td>3.227</td>
        </tr>
        <tr>
            <td>segtree.c</td>
            <td>7</td>
            <td><strong>0.723</strong></td>
            <td>90</td>
            <td>1.977</td>
            <td>15</td>
            <td>0.855</td>
            <td>6</td>
            <td>0.723</td>
            <td><strong>6</strong></td>
            <td>0.801</td>
        </tr>
        <tr>
            <td>segtree.cpp</td>
            <td><strong>6</strong></td>
            <td>0.867</td>
            <td>77</td>
            <td>1.852</td>
            <td>15</td>
            <td>1.094</td>
            <td>7</td>
            <td>0.871</td>
            <td>9</td>
            <td>0.934</td>
        </tr>
    </tbody>
</table>
