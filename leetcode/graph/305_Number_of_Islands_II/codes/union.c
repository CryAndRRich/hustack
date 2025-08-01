#define MAX 10000

int find(int* parent, int x) {
    if (parent[x] != x)
        parent[x] = find(parent, parent[x]);
    return parent[x];
}

int union_set(int* parent, int* rank, int x, int y) {
    int xr = find(parent, x);
    int yr = find(parent, y);
    if (xr == yr) return 0;
    if (rank[xr] < rank[yr]) parent[xr] = yr;
    else if (rank[xr] > rank[yr]) parent[yr] = xr;
    else { parent[yr] = xr; rank[xr]++; }
    return 1;
}

int* numIslands2(int m, int n, int** positions, int positionsSize, int* positionsColSize, int* returnSize) {
    int* res = malloc(sizeof(int) * positionsSize);
    int parent[MAX], rank[MAX];
    for (int i = 0; i < m * n; i++) parent[i] = -1, rank[i] = 0;
    int count = 0;

    int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};

    for (int k = 0; k < positionsSize; k++) {
        int r = positions[k][0], c = positions[k][1], idx = r * n + c;
        if (parent[idx] != -1) {
            res[k] = count;
            continue;
        }
        parent[idx] = idx;
        count++;

        for (int d = 0; d < 4; d++) {
            int nr = r + dirs[d][0], nc = c + dirs[d][1];
            int nidx = nr * n + nc;
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && parent[nidx] != -1) {
                if (union_set(parent, rank, idx, nidx)) count--;
            }
        }
        res[k] = count;
    }

    *returnSize = positionsSize;
    return res;
}
