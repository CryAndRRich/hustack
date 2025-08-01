typedef struct NodeCol {
    struct TreeNode* node;
    int col;
} NodeCol;

int** verticalOrder(struct TreeNode* root, int* returnSize, int** returnColumnSizes) {
    if (!root) {
        *returnSize = 0;
        *returnColumnSizes = NULL;
        return NULL;
    }

    NodeCol queue[200];
    int front = 0, rear = 0;
    queue[rear++] = (NodeCol){root, 0};

    int colMin = 0, colMax = 0;
    int* colCount = calloc(401, sizeof(int));
    int** colData = calloc(401, sizeof(int*));

    while (front < rear) {
        NodeCol nc = queue[front++];
        struct TreeNode* node = nc.node;
        int col = nc.col;

        int idx = col + 200;
        if (!colData[idx]) colData[idx] = malloc(100 * sizeof(int));
        colData[idx][colCount[idx]++] = node -> val;

        if (node -> left) queue[rear++] = (NodeCol){node -> left, col - 1};
        if (node -> right) queue[rear++] = (NodeCol){node -> right, col + 1};

        if (col < colMin) colMin = col;
        if (col > colMax) colMax = col;
    }

    int totalCols = colMax - colMin + 1;
    int** result = malloc(totalCols * sizeof(int*));
    *returnColumnSizes = malloc(totalCols * sizeof(int));
    *returnSize = totalCols;

    for (int i = 0; i < totalCols; ++i) {
        result[i] = colData[i + colMin + 200];
        (*returnColumnSizes)[i] = colCount[i + colMin + 200];
    }

    free(colCount);
    free(colData);
    return result;
}
