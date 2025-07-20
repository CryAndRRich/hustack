long long maximumBooks(int* books, int booksSize) {
    long long* dp = calloc(booksSize, sizeof(long long));
    int* stack = malloc(booksSize * sizeof(int));
    int top = -1;
    long long res = 0;

    for (int i = 0; i < booksSize; ++i) {
        while (top >= 0 && books[stack[top]] >= books[i] - (i - stack[top])) {
            --top;
        }

        if (top < 0) {
            int len = books[i] < i + 1 ? books[i] : i + 1;
            int first = books[i] - len + 1;
            dp[i] = ((long long)(books[i] + first) * len) / 2;
        } else {
            int j = stack[top];
            int len = i - j;
            int first = books[i] - len + 1;
            dp[i] = dp[j] + ((long long)(books[i] + first) * len) / 2;
        }

        stack[++top] = i;
        if (dp[i] > res) res = dp[i];
    }

    free(dp);
    free(stack);
    return res;
}
