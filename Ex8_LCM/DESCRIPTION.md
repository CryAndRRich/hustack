## Description
Given $n$ positive integers $a_1, a_2, ..., a_n$. Compute the smallest positive integer $N$ such that $N$ is divisible by $a_1, a_2, ..., a_n$

**Input:**
- **Line 1**: Contain a positive integer $n$ $(1 \leq n \leq 100000)$
- **Line 2**: Contain $a_1, a_2, ..., a_n$ $(1 \leq a_i \leq 1000)$

**Output:**
- The value of $N$ mod $1000000000$

## Approach 1: Brute Force
Each number is factorized by trying divisors sequentially (starting from 2) until the number is reduced to 1.
It is more precisely referred to as **trial division** for prime factorization

- **Time Complexity**: $O(n)$
- **Space Complexity**: $O(1)$

## Approach 2: Sieve of Eratosthenes
Precompute all the prime numbers up to 1000 using a sieve (since every $a_i$ is $\leq 1000$). Then, for each prime $p$ in that list, determine the maximum exponent $e$ such that $p^e$ divides at least one of the numbers in the input array. The final answer is the product over all such primes.

- **Time Complexity**: $O(n)$
- **Space Complexity**: $O(1)$

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
            <td>brute.c</td>
            <td><strong>6</strong></td>
            <td>0.781</td>
            <td><strong>6</strong></td>
            <td>0.801</td>
            <td>21</td>
            <td>0.898</td>
            <td>110</td>
            <td>1.215</td>
            <td><strong>6</strong></td>
            <td><strong>0.676</strong></td>
        </tr>
        <tr>
            <td>eratosthenes.c</td>
            <td><strong>6</strong></td>
            <td><strong>0.777</strong></td>
            <td><strong>6</strong></td>
            <td><strong>0.742</strong></td>
            <td><strong>10</strong></td>
            <td><strong>0.789</strong></td>
            <td><strong>44</strong></td>
            <td><strong>1.121</strong></td>
            <td><strong>6</strong></td>
            <td>0.777</td>
        </tr>
    </tbody>
</table>
