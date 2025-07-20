## Description
Given a sequence of integers $a_1, a_2, \dots, a_n$. A pair $(i, j)$ is called an **inversion** if $i < j$ and $a_i > a_j$. Compute the total number $Q$ of inversions in the sequence.

**Input**:
- **Line 1:** A positive integer $n$ $(1 \leq n \leq 10^6)$.
- **Line 2:** $n$ integers $a_1, a_2, \dots, a_n$ where $0 \le a_i \le 10^6$.

**Output**: A single integer- the value of $Q$ modulo $10^9 + 7$.

## Approach 1: Merge Sort
- Recursively split the array into two halves until single-element arrays.
- When merging sorted halves $L$ and $R$, if $L[i] <= R[j]$ then no new inversions; otherwise, $L[i] > R[j]$ implies all remaining elements in $L$ from index $i$ onward form inversions with $R[j]$.
- Accumulate these counts while merging.
- Sum inversion counts from left half, right half, and cross inversions, taking modulo $10^9+7$ at each step.

**Time Complexity:**
- Each merge step processes $O(n)$ elements.
- Recurrence $T(n) = 2T(n/2) + O(n)$ gives $O(n \log n)$.  

**Space Complexity:** $O(n)$
- Auxiliary array for merging: $O(n)$.
- Recursion depth: $O(\log n)$.

## Approach 2: Fenwick Tree (Binary Indexed Tree)
- Map values $a_i$ into the range $[1..n]$ by sorting unique values and using their rank.
- Initialize a Fenwick Tree $BIT$ of size $n$ with all zeros.
- For each index $i$ from $n$ down to 1:
    - Query $BIT.query(a_i - 1)$ to count how many values less than $a_i$ have been seen (i.e., to the right).
    - Add this count to the inversion total.
    - Update $BIT.update(a_i, 1)$ to mark $a_i$ as seen.
- Apply modulo $10^9+7$ on updates to the total count.

**Time Complexity:** $O(n \log n)$
- Coordinate compression: $O(n \log n)$.
- $n$ BIT operations: each $O(\log n)$.

**Space Complexity:** $O(n)$
- Fenwick Tree array: $O(n)$.
- Compressed value array and auxiliary structures: $O(n)$.
