#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

int helper(const char* s, int* idx) {
    int stack[10000], top = -1;
    int num = 0;
    char sign = '+';

    while (s[*idx]) {
        char c = s[(*idx)++];

        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        }

        if (c == '(') {
            num = helper(s, idx);
        }

        if ((!isdigit(c) && c != ' ') || s[*idx] == '\0') {
            if (sign == '+') stack[++top] = num;
            else if (sign == '-') stack[++top] = -num;
            else if (sign == '*') {
                stack[top] = stack[top] * num;
            } else if (sign == '/') {
                stack[top] = stack[top] / num;
            }
            sign = c;
            num = 0;
        }

        if (c == ')') break;
    }

    int sum = 0;
    for (int i = 0; i <= top; ++i) sum += stack[i];
    return sum;
}

int calculate(char* s) {
    int idx = 0;
    return helper(s, &idx);
}
