#include <stdbool.h>

bool confusingNumber(int n) {
    int map[10];
    for (int i = 0; i < 10; ++i) map[i] = -1;
    map[0] = 0; map[1] = 1; map[6] = 9; map[8] = 8; map[9] = 6;

    int original = n, rotated = 0;
    while (n > 0) {
        int d = n % 10;
        if (map[d] == -1) return false;
        rotated = rotated * 10 + map[d];
        n /= 10;
    }
    return rotated != original;
}
