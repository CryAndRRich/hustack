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
