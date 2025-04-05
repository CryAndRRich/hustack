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
            <td>brute.py</td>
            <td>43</td>
            <td>3.145</td>
            <td>41</td>
            <td>3.262</td>
            <td>45</td>
            <td>3.148</td>
            <td>48</td>
            <td>3.203</td>
            <td>42</td>
            <td>3.188</td>
        </tr>
        <tr>
            <td>brute.c</td>
            <td><strong>6</strong></td>
            <td><strong>0.746</strong></td>
            <td><strong>6</strong></td>
            <td><strong>0.789</strong></td>
            <td>7</td>
            <td>0.766</td>
            <td><strong>5</strong></td>
            <td>0.797</td>
            <td><strong>6</strong></td>
            <td><strong>0.844</strong></td>
        </tr>
        <tr>
            <td>brute.cpp</td>
            <td><strong>6</strong></td>
            <td>0.914</td>
            <td>7</td>
            <td>0.852</td>
            <td>8</td>
            <td><strong>0.742</strong></td>
            <td>8</td>
            <td><strong>0.793</strong></td>
            <td>8</td>
            <td>0.879</td>
        </tr>
        <tr>
            <td>bfs.py</td>
            <td>44</td>
            <td>3.395</td>
            <td>26</td>
            <td>3.41</td>
            <td>28</td>
            <td>3.477</td>
            <td>46</td>
            <td>3.449</td>
            <td>51</td>
            <td>3.695</td>
        </tr>
        <tr>
            <td>bfs.c</td>
            <td>7</td>
            <td>1.715</td>
            <td>7</td>
            <td>1.746</td>
            <td><strong>6</strong></td>
            <td>1.719</td>
            <td>7</td>
            <td>1.746</td>
            <td><strong>6</strong></td>
            <td>1.715</td>
        </tr>
        <tr>
            <td>bfs.cpp</td>
            <td>9</td>
            <td>1.711</td>
            <td>7</td>
            <td>1.723</td>
            <td>9</td>
            <td>1.75</td>
            <td>9</td>
            <td>1.711</td>
            <td>11</td>
            <td>1.672</td>
        </tr>
    </tbody>
</table>
