## Description
A rectangle of size $n \times m$ is divided into $1 \times 1$ square cells, each colored either black or white. The rectangle is represented by an $n \times m$ matrix $A$, where $A(i,j)=1$ means that the cell in row $i$ and column $j$ is black, and $A(i,j)=0$ means that the cell is white (the rows and columns are numbered starting from $1$). A subrectangle of the board is defined as a rectangle whose top-left cell is $(1, 1)$. Determine the subrectangle of the given board that contains only black cells and has the maximum area.

**Input:**
- **Line $1$:** Two positive integers $n$ and $m$ $(1 \leq n,m \leq 50)$
- **Lines $2$ to $n+1$:** Each line contains the elements of a row of matrix $A$

**Output:** Print the area of the largest subrectangle found

## Approach: Prefix Sum
For each cell $(i, j)$ ($1$-indexed), compute:
$$S[i][j] = A[i][j] + S[i-1][j] + S[i][j-1] - S[i-1][j-1]$$
This makes $S[i][j]$ the sum of the submatrix from $(1, 1)$ to $(i, j)$. Iterate over all possible bottom-right corners $(i, j)$ of subrectangles starting at $(1,1)$. For a subrectangle with area $i \times j$, if $S[i][j] == i * j$ then all cells are black

**Time Complexity:** $O(n \times m)$

**Space Complexity:** $O(n \times m)$


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
            <td>prefix.py</td>
            <td>42</td>
            <td>3.18</td>
            <td>443</td>
            <td>14.77</td>
            <td>TLE</td>
            <td></td>
            <td>64</td>
            <td>3.5</td>
            <td>41</td>
            <td>3.145</td>
        </tr>
        <tr>
            <td>prefix.c</td>
            <td><strong>5</strong></td>
            <td><strong>0.723</strong></td>
            <td><strong>31</strong></td>
            <td><strong>2.633</strong></td>
            <td><strong>128</strong></td>
            <td><strong>8.348</strong></td>
            <td><strong>8</strong></td>
            <td>0.977</td>
            <td><strong>6</strong></td>
            <td><strong>0.836</strong></td>
        </tr>
        <tr>
            <td>prefix.cpp</td>
            <td>8</td>
            <td>0.906</td>
            <td>73</td>
            <td>2.809</td>
            <td>188</td>
            <td>8.473</td>
            <td><strong>8</strong></td>
            <td><strong>0.828</strong></td>
            <td>17</td>
            <td>0.844</td>
        </tr>
    </tbody>
</table>
