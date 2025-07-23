#include <stdio.h>
#include <stdbool.h>

int rotate_digit(int d) {
    switch (d) {
        case 0: return 0;
        case 1: return 1;
        case 6: return 9;
        case 8: return 8;
        case 9: return 6;
        default: return -1;
    }
}

bool is_confusing(int x) {
    int original = x, rotated = 0;
    while (x > 0) {
        int d = x % 10;
        int r = rotate_digit(d);
        if (r == -1) return false; 
        rotated = rotated * 10 + r;
        x /= 10;
    }
    return rotated != original;
}

void dfs(int curr, int n, int *count) {
    if (curr > n) return;
    if (curr != 0 && is_confusing(curr)) {
        (*count)++;
    }
    int digits[] = {0, 1, 6, 8, 9};
    for (int i = 0; i < 5; i++) {
        int d = digits[i];
        if (curr == 0 && d == 0) continue;
        int next = curr * 10 + d;
        if (next <= n)
            dfs(next, n, count);
    }
}

int confusingNumberII(int n) {
    int count = 0;
    dfs(0, n, &count);
    return count;
}