#include <stdio.h>
#include <string.h>

#define MAX 1000005

char stack[MAX];
int top = -1;

int check_pair(char open, char close) {
    return (open == '(' && close == ')') ||
           (open == '[' && close == ']') ||
           (open == '{' && close == '}');
}

int check_correct(const char *s) {
    for (int i = 0; s[i]; ++i) {
        char c = s[i];
        if (c == '(' || c == '[' || c == '{') {
            stack[++top] = c;
        } else {
            if (top < 0 || !check_pair(stack[top--], c)) {
                return 0;
            }
        }
    }
    return top == -1;
}

int main() {
    char s[MAX];
    scanf("%s", s);
    printf("%d\n", check_correct(s));
    return 0;
}
