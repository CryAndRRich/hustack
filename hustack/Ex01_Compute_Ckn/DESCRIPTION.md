## Description
Given two positive integers $k$ and $n$. 

Compute $C(k,n)$ which is the number of ways to select $k$ objects from a given set of $n$ objects.

**Input**: Two positive integers $k$ and $n$ $(1 \leq k,n \leq 999)$

**Output**: The value $C(k,n)$ modulo $10^9+7$

## Approach 1: Recursion + Memoization
We compute the value $C(k,n)$ by using recursion. The idea is to break down the problem into smaller sub-problems using Pascal's identity:
$$C(k,n)=C(k,n−1)+C(k−1,n−1)$$
Since direct recursion would lead to redundant computations, memoization is used to store already computed values in a 2D array to avoid recalculating them

### Time Complexity:
With memoization, each subproblem $C(n,k)$ is computed only once, leading to a time complexity of $O(nk)$
### Space Complexity:
The 2D array stores computed values, leading to a space complexity of $O(nk)$

## Approach 2: Inverse Module
Instead of using recursion or a DP table, the formula for $C(n,k)$ is computed directly using:
$$C(n,k) = \frac{n!}{k!(n-k)!} \mod MOD$$
However, direct division is not possible in modular arithmetic. To handle this, the modular inverse of each factorial is used. Specifically, Fermat’s Little Theorem states:
$$a^{-1} \equiv a^{MOD-2} \mod MOD$$
Thus, the modular inverse of $k!$ and $(n-k)!$ can be computed using fast exponentiation (binary exponentiation).

### Time Complexity:
- The loop runs $O(k)$ times
- Each modular inverse calculation takes $O(log MOD)$ using binary exponentiation

Overall time complexity: $O(k + logMOD)$
### Space Complexity:
Only a few extra variables are used, leading to a space complexity of $O(1)$
