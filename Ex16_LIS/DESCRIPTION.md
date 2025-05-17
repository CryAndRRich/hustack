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
