#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* stripLeadingZeros(const char* str) {
    while (*str == '0' && *(str + 1) != '\0') {
        str++;
    }
    return strdup(str);
}

char* bigAdd(const char* num1, const char* num2) {
    int len1 = strlen(num1);
    int len2 = strlen(num2);
    int maxLen = (len1 > len2) ? len1 : len2;
    int carry = 0;
    char* result = (char*) malloc(maxLen + 2); 
    result[maxLen + 1] = '\0';
    int i = len1 - 1, j = len2 - 1, k = maxLen;
    
    while (i >= 0 || j >= 0 || carry) {
        int digit1 = (i >= 0) ? num1[i] - '0' : 0;
        int digit2 = (j >= 0) ? num2[j] - '0' : 0;
        int sum = digit1 + digit2 + carry;
        carry = sum / 10;
        result[k] = (sum % 10) + '0';
        i--; 
        j--; 
        k--;
    }
    return strdup(result + k + 1);
}

char* bigSubtract(const char* num1, const char* num2) {
    int len1 = strlen(num1);
    int len2 = strlen(num2);
    char* result = (char*) malloc(len1 + 1);
    result[len1] = '\0';
    int carry = 0;
    int i = len1 - 1, j = len2 - 1, k = len1 - 1;
    
    while (i >= 0) {
        int digit1 = num1[i] - '0';
        int digit2 = (j >= 0) ? num2[j] - '0' : 0;
        int sub = digit1 - digit2 - carry;
        if (sub < 0) {
            sub += 10;
            carry = 1;
        } else {
            carry = 0;
        }
        result[k] = sub + '0';
        i--; 
        j--; 
        k--;
    }
    char* stripped = stripLeadingZeros(result);
    free(result);
    return stripped;
}

char* multiplyByPowerOf10(const char* num, int power) {
    int len = strlen(num);
    char* result = (char*) malloc(len + power + 1);
    strcpy(result, num);
    for (int i = 0; i < power; i++) {
        result[len + i] = '0';
    }
    result[len + power] = '\0';
    return result;
}

char* multiplySingleDigits(const char* a, const char* b) {
    int digit1 = a[0] - '0';
    int digit2 = b[0] - '0';
    int prod = digit1 * digit2;
    char buffer[3];
    sprintf(buffer, "%d", prod);
    return strdup(buffer);
}

char* karatsuba(const char* x, const char* y) {
    char* xStripped = stripLeadingZeros(x);
    char* yStripped = stripLeadingZeros(y);
    int lenX = strlen(xStripped);
    int lenY = strlen(yStripped);
    
    if (lenX == 1 && lenY == 1) {
        char* product = multiplySingleDigits(xStripped, yStripped);
        free(xStripped);
        free(yStripped);
        return product;
    }
    
    int n = (lenX > lenY) ? lenX : lenY;
    if (n % 2 != 0) {
        n++;
    }
    char* X = (char*) malloc(n + 1);
    char* Y = (char*) malloc(n + 1);
    
    memset(X, '0', n - lenX);
    strcpy(X + (n - lenX), xStripped);
    X[n] = '\0';
    
    memset(Y, '0', n - lenY);
    strcpy(Y + (n - lenY), yStripped);
    Y[n] = '\0';
    
    free(xStripped);
    free(yStripped);
    
    int m = n / 2;
    char* X_high = strndup(X, n - m);
    char* X_low  = strdup(X + n - m);
    char* Y_high = strndup(Y, n - m);
    char* Y_low  = strdup(Y + n - m);
    
    char* z2 = karatsuba(X_high, Y_high);
    char* z0 = karatsuba(X_low, Y_low);
    
    char* sum1 = bigAdd(X_high, X_low);
    char* sum2 = bigAdd(Y_high, Y_low);
    char* z1 = karatsuba(sum1, sum2);
    
    char* temp = bigAdd(z2, z0);
    char* z1_final = bigSubtract(z1, temp);
    
    char* z2_shift = multiplyByPowerOf10(z2, 2 * m);
    char* z1_shift = multiplyByPowerOf10(z1_final, m);
    char* temp2 = bigAdd(z2_shift, z1_shift);
    char* result = bigAdd(temp2, z0);
    
    free(X);
    free(Y);
    free(X_high);
    free(X_low);
    free(Y_high);
    free(Y_low);
    free(sum1);
    free(sum2);
    free(z2);
    free(z0);
    free(z1);
    free(temp);
    free(z1_final);
    free(z2_shift);
    free(z1_shift);
    free(temp2);
    
    char* finalResult = stripLeadingZeros(result);
    free(result);
    return finalResult;
}

int main() {
    char a[11000], b[11000];
    scanf("%s", a);
    scanf("%s", b);
    
    char* result = karatsuba(a, b);
    printf("%s\n", result);
    free(result);
    return 0;
}
