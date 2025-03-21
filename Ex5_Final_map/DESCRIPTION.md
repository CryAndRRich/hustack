## Description
The HUST center is developing a system for displaying vehicles on a digital map. Each vehicle's location is given by a pair of coordinates (longitude and latitude), for example: $(x, y) = (105.84399, 20.99654)$. The initial list contains $N$ vehicles ($N > 0$ and $N \leq 50000$). When the app starts, rendering the entire map and all vehicles takes a long time, which frustrates users. Therefore, the programmer decides to optimize the display by only showing vehicles whose coordinates fall strictly within the current viewport (vehicles on the boundary are ignored). The viewport is defined as a rectangle given by the coordinates $(x, y)$ of its top-left and bottom-right corners. Note that Vietnam spans approximately from latitude 7 to 30 and longitude 100 to 110.

As users interact with the application by panning and zooming (in and out), the viewport changes continuously. Consequently, the programmer needs to repeatedly check which vehicles are within the viewport in order to add or remove them from the display

**Input**: 
- **Line 1**: Two numbers representing the number of vehicles $N$ ($N > 0$ and $N \leq 50000$) and the number of queries $M$ ($M > 0$ and $M \leq 5000$).
- **Next $N$ lines**: Each line contains 2 real numbers (separated by a space) representing the coordinates of a vehicle (longitude and latitude, in that order, within the specified ranges).
- **Following $M$ lines**: Each line contains 4 real numbers, corresponding to the coordinates of the top-left and bottom-right corners of the viewport.

**Output**:
For each query, output the number of vehicles that lie strictly within the viewport (vehicles exactly on the boundary are not counted).

## Approach 1: Brute Force
In this approach, for each query we simply iterate over all vehicles and check whether each vehicle’s coordinates are strictly within the viewport. This is a direct and intuitive method that requires no preprocessing.

### Time Complexity:
- For each query, we check all $N$ vehicles.
- Overall time complexity is $O(M.N)$.

### Space Complexity:
- We store the coordinates for all vehicles, leading to a space complexity of $O(N)$.

## Approach 2: Binary Search
This approach leverages binary search after preprocessing the list of vehicles based on their x-coordinates. By sorting the vehicles, we can quickly identify the subset that falls within the horizontal range of the viewport and then filter those by their y-coordinate.

### Time Complexity:
- Sorting the vehicles takes $O(N \log N)$.
- Binary search takes $O(\log N)$.
- Overall time complexity is $O(N \log N)$.

### Space Complexity:
- We store the coordinates and an auxiliary index array, resulting in a space complexity of $O(N)$.

## Approach 3: KD-Tree
We use a **KD-Tree** for range queries, which is a spatial data structure that partitions the plane recursively by alternating between the x- and y-dimensions. For each query (given by left, top, right, and bottom boundaries), perform a range search in the KD-Tree:
- At each node, check if the current vehicle lies strictly within the viewport.
- Depending on the splitting axis and the node’s coordinate, decide whether to search the left and/or right subtree.

### Time Complexity:
- It takes $O(N \log N)$ time to build the KD-Tree.
- Average-case $O(\sqrt{N}+k)$ per query.

### Space Complexity:
- The KD-Tree requires $O(N)$ space.

## Result:
<table style="text-align: center;">
    <thead>
        <tr>
            <th style="text-align: center;">File</th>
            <th colspan="2" style="text-align: center;">brute.c</th>
            <th colspan="2" style="text-align: center;">binary.c</th>
            <th colspan="2" style="text-align: center;">kdtree.c</th>
        </tr>
        <tr>
            <th></th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
    </thead>
    <tbody>
        <tr>
            <td>Test 1</td>
            <td>TLE</td>
            <td></td>
            <td>727</td>
            <td><strong>1.715</strong></td>
            <td><strong>239</strong></td>
            <td>4.938</td>
        </tr>
        <tr>
            <td>Test 2</td>
            <td>322</td>
            <td><strong>1.012</strong></td>
            <td>137</td>
            <td>1.184</td>
            <td><strong>97</strong></td>
            <td>2.348</td>
        </tr>
        <tr>
            <td>Test 3</td>
            <td>943</td>
            <td><strong>1.422</strong></td>
            <td>478</td>
            <td>1.586</td>
            <td><strong>190</strong></td>
            <td>4.133</td>
        </tr>
        <tr>
            <td>Test 4</td>
            <td>106</td>
            <td><strong>0.895</strong></td>
            <td>59</td>
            <td>1.012</td>
            <td><strong>57</strong></td>
            <td>1.516</td>
        </tr>
        <tr>
            <td>Test 5</td>
            <td><strong>6</strong></td>
            <td><strong>0.641</strong></td>
            <td><strong>6</strong></td>
            <td>0.719</td>
            <td><strong>6</strong></td>
            <td>0.715</td>
        </tr>
        <tr>
            <td>Test 6</td>
            <td>7</td>
            <td>0.898</td>
            <td><strong>6</strong></td>
            <td><strong>0.734</strong></td>
            <td><strong>6</strong></td>
            <td>0.746</td>
        </tr>
        <tr>
            <td>Test 7</td>
            <td>642</td>
            <td><strong>1.207</strong></td>
            <td>281</td>
            <td>1.406</td>
            <td><strong>141</strong></td>
            <td>3.277</td>
        </tr>
        <tr>
            <td>Test 8</td>
            <td>TLE</td>
            <td></td>
            <td>721</td>
            <td><strong>1.781</strong></td>
            <td><strong>240</strong></td>
            <td>4.949</td>
        </tr>
        <tr>
            <td>Test 9</td>
            <td><strong>5</strong></td>
            <td>0.816</td>
            <td>6</td>
            <td><strong>0.711</strong></td>
            <td>6</td>
            <td>0.828</td>
        </tr>
        <tr>
            <td>Test 10</td>
            <td>7</td>
            <td><strong>0.684</strong></td>
            <td><strong>5</strong></td>
            <td>0.73</td>
            <td>6</td>
            <td>0.711</td>
        </tr>
        <tr>
            <td>Test 11</td>
            <td>7</td>
            <td>0.727</td>
            <td><strong>6</strong></td>
            <td>0.742</td>
            <td>8</td>
            <td><strong>0.711</strong></td>
        </tr>
        <tr>
            <td>Test 12</td>
            <td><strong>6</strong></td>
            <td>0.73</td>
            <td><strong>6</strong></td>
            <td>0.719</td>
            <td><strong>6</strong></td>
            <td><strong>0.711</strong></td>
        </tr>
        <tr>
            <td>Test 13</td>
            <td>TLE</td>
            <td></td>
            <td>703</td>
            <td><strong>1.941</strong></td>
            <td><strong>250</strong></td>
            <td>4.98</td>
        </tr>
    </tbody>
</table>
