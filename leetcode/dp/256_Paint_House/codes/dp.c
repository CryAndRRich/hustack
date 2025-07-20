int minCost(int** costs, int costsSize, int* costsColSize) {
    if (costsSize == 0) return 0;

    for (int i = costsSize - 2; i >= 0; i--) {
        int r = costs[i][0] + fmin(costs[i + 1][1], costs[i + 1][2]);
        int g = costs[i][1] + fmin(costs[i + 1][0], costs[i + 1][2]);
        int b = costs[i][2] + fmin(costs[i + 1][0], costs[i + 1][1]);

        costs[i][0] = r;
        costs[i][1] = g;
        costs[i][2] = b;
    }

    int minCost = costs[0][0];
    if (costs[0][1] < minCost) minCost = costs[0][1];
    if (costs[0][2] < minCost) minCost = costs[0][2];
    return minCost;
}
