## Description
There are two jugs with capacities $a$ and $b$ litres, respectively, and an unlimited water supply. 
Given a positive integer $c$, the task is to obtain exactly $c$ litres of water using the jugs. If it is not possible, output $-1$

**Input**: 
Three positive integers $a, b, c$ $(1 \leq a, b, c \leq 900)$

**Output**: 
The minimum number of steps to get exactly $c$ litres, or $-1$ if no solution exists

## Approach 1: Brute Force (Simulation)
This approach simulates the water pouring process using a set of operations:
- **Fill** a jug completely
- **Empty** a jug
- **Pour** water from one jug to the other until either the source is empty or the target jug is full

The algorithm employs these operations repeatedly while checking if either jug contains exactly $c$ litres. It also uses the **gcd (greatest common divisor)** to verify whether the desired quantity is achievable (since $c$ must be a multiple of $gcd(a, b)$).

### Time Complexity:  
The simulation performs a series of constant-time operations per step. Although the number of steps can vary, in the worst case it is bounded by a function of $a$ and $b$

### Space Complexity:  
Only a fixed number of variables are used, resulting in a constant space usage, i.e., $O(1)$

## Approach 2: Breadth-First Search (BFS)
In the BFS approach, each state is defined by the current amount of water in the two jugs. The allowed moves (or transitions) are:
- **Fill** either jug
- **Empty** either jug
- **Pour** water from one jug to the other

Using BFS, the algorithm explores all possible states starting from $(0, 0)$ until it finds a state where either jug contains exactly $c$ litres. A 2D array (or similar structure) keeps track of visited states to avoid redundant work.

### Time Complexity:  
In the worst case, the BFS will explore up to $(a+1) \times (b+1)$ states, leading to a complexity of $O(ab)$

### Space Complexity:  
The space is dominated by the queue and the visited states array, both of which have a size of $O(ab)$
