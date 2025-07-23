#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int pow5(int exp) {
    int res = 1;
    while (exp--) res *= 5;
    return res;
}

int calc(const char* s, int idx, int* digits, int* map, int half, int odd, int len) {
    int num = s[idx] - '0';
    int index = (num > 8) ? 4 : (num > 6) ? 3 : (num > 1) ? 2 : (num == 1) ? 1 : 0;

    if (idx == len - 1) return index + (digits[index] == num);

    int smaller = (idx == 0 ? index - 1 : index);
    int res = smaller * pow5(len - idx - 1);

    if (idx < half)
        res -= smaller * pow5(half - idx - 1) * (odd ? 3 : 1);
    if (idx == half && odd)
        res -= (num > 8 ? 3 : (num > 1 ? 2 : (num > 0 ? 1 : 0)));
    if (digits[index] == num)
        res += calc(s, idx + 1, digits, map, half, odd, len);
    return res;
}

int confusingNumberII(int n) {
    int digits[5] = {0, 1, 6, 8, 9};
    int map[10] = {0}; 
    map[0] = 0; map[1] = 1; map[6] = 9; map[8] = 8; map[9] = 6;

    char s[20];
    sprintf(s, "%d", n);
    int len = strlen(s);
    int half = len / 2;
    int odd = len % 2;
    int ans = 2;

    for (int l = 2; l < len; l++) {
        int tmp = pow5(l - 1);
        int sub = pow5(l / 2 - 1) * (l % 2 ? 3 : 1);
        ans += 4 * (tmp - sub);
    }

    ans += calc(s, 0, digits, map, half, odd, len);

    int first_half_num = 0;
    for (int i = half - 1; i >= 0; i--) {
        int d = s[i] - '0';
        if (map[d] == 0 && d != 0) return ans;
        first_half_num = first_half_num * 10 + map[d];
    }

    if (odd) {
        if (s[half] == '0' || s[half] == '1' || s[half] == '8') {
            int tail = atoi(s + half + 1);
            if (tail >= first_half_num) ans--;
        }
    } else {
        int tail = atoi(s + half);
        if (tail >= first_half_num) ans--;
    }
    return ans;
}
