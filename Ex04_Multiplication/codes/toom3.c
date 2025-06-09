#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char* stripLeadingZeros(const char* str) {
    while (*str == '0' && *(str + 1) != '\0') {
        str++;
    }
    return strdup(str);
}

char* stripLeadingZerosSigned(const char* str) {
    if(str[0] == '-') {
        const char* p = str + 1;
        while (*p == '0' && *(p + 1) != '\0')
            p++;
        char* res = (char*) malloc(strlen(p) + 2);
        res[0] = '-';
        strcpy(res + 1, p);
        return res;
    } else {
        return stripLeadingZeros(str);
    }
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
    char* ret = strdup(result + k + 1);
    free(result);
    return ret;
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

int compareAbs(const char* a, const char* b) {
    while (*a == '0' && *(a + 1) != '\0') a++;
    while (*b == '0' && *(b + 1) != '\0') b++;
    int lenA = strlen(a), lenB = strlen(b);
    if (lenA != lenB)
        return (lenA > lenB) ? 1 : -1;
    return strcmp(a, b);
}

char* bigAddSigned(const char* a, const char* b) {
    if(a[0] == '-' && b[0] == '-') {
        char* res = bigAdd(a + 1, b + 1);
        if(strcmp(res, "0") != 0) {
            char* res2 = (char*) malloc(strlen(res) + 2);
            res2[0] = '-';
            strcpy(res2 + 1, res);
            free(res);
            return res2;
        }
        return res;
    }
    if(a[0] == '-' && b[0] != '-') {
        int cmp = compareAbs(a + 1, b);
        if(cmp == 0) return strdup("0");
        if(cmp > 0) {
            char* res = bigSubtract(a + 1, b);
            if(strcmp(res, "0") != 0) {
                char* res2 = (char*) malloc(strlen(res) + 2);
                res2[0] = '-';
                strcpy(res2 + 1, res);
                free(res);
                return res2;
            }
            return res;
        } else {
            return bigSubtract(b, a + 1);
        }
    }
    if(a[0]!='-' && b[0] == '-') {
        int cmp = compareAbs(a, b + 1);
        if(cmp == 0) return strdup("0");
        if(cmp > 0) {
            return bigSubtract(a, b + 1);
        } else {
            char* res = bigSubtract(b + 1, a);
            if(strcmp(res, "0") != 0) {
                char* res2 = (char*) malloc(strlen(res) + 2);
                res2[0] = '-';
                strcpy(res2 + 1, res);
                free(res);
                return res2;
            }
            return res;
        }
    }
    return bigAdd(a, b);
}

char* bigSubtractSigned(const char* a, const char* b) {
    if(b[0] == '-') {
        return bigAddSigned(a, b + 1);
    }
    if(a[0] == '-') {
        char* res = bigAdd(a + 1, b);
        if(strcmp(res, "0") != 0) {
            char* res2 = (char*) malloc(strlen(res) + 2);
            res2[0] = '-';
            strcpy(res2 + 1, res);
            free(res);
            return res2;
        }
        return res;
    }
    int cmp = compareAbs(a, b);
    if(cmp == 0) return strdup("0");
    if(cmp > 0) {
        return bigSubtract(a, b);
    } else {
        char* res = bigSubtract(b, a);
        if(strcmp(res, "0") != 0) {
            char* res2 = (char*) malloc(strlen(res) + 2);
            res2[0] = '-';
            strcpy(res2 + 1, res);
            free(res);
            return res2;
        }
        return res;
    }
}

char* bigMultiplyByInt(const char* num, int multiplier) {
    int sign = 1;
    if(num[0] == '-') {
        sign = -1;
        num++;
    }
    int len = strlen(num);
    int carry = 0;
    char* result = (char*) malloc(len + 10);
    result[len] = '\0';
    int pos = len - 1;
    while(pos >= 0) {
        int digit = num[pos]-'0';
        int prod = digit * multiplier + carry;
        carry = prod / 10;
        result[pos] = (prod % 10) + '0';
        pos--;
    }
    while(carry) {
        int currentLen = strlen(result);
        char* newResult = (char*) malloc(currentLen + 2);
        newResult[0] = (carry % 10) + '0';
        strcpy(newResult+1, result);
        free(result);
        result = newResult;
        carry /= 10;
    }
    char* resStr = stripLeadingZeros(result);
    free(result);
    if(sign < 0 && strcmp(resStr, "0") != 0) {
        char* res2 = (char*) malloc(strlen(resStr) + 2);
        res2[0] = '-';
        strcpy(res2 + 1, resStr);
        free(resStr);
        return res2;
    }
    return resStr;
}

char* bigDivideByInt(const char* num, int divisor) {
    int sign = 1;
    if(num[0] == '-') {
        sign = -1;
        num++;
    }
    int len = strlen(num);
    char* result = (char*) malloc(len + 1);
    int idx = 0;
    int temp = 0;
    for (int i = 0; i < len; i++){
        temp = temp * 10 + (num[i] - '0');
        int q = temp / divisor;
        result[idx++] = q + '0';
        temp = temp % divisor;
    }
    result[idx] = '\0';
    char* resStr = stripLeadingZeros(result);
    free(result);
    if(sign < 0 && strcmp(resStr, "0") != 0) {
        char* res2 = (char*) malloc(strlen(resStr)+2);
        res2[0] = '-';
        strcpy(res2 + 1, resStr);
        free(resStr);
        return res2;
    }
    return resStr;
}

char* multiplySingleDigits(const char* a, const char* b) {
    int digit1 = a[0] - '0';
    int digit2 = b[0] - '0';
    int prod = digit1 * digit2;
    char buffer[3];
    sprintf(buffer, "%d", prod);
    return strdup(buffer);
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

char* toom3(const char* x, const char* y) {
    char* xStripped = stripLeadingZerosSigned(x);
    char* yStripped = stripLeadingZerosSigned(y);
    
    const char* xAbs = (xStripped[0] == '-') ? xStripped+1 : xStripped;
    const char* yAbs = (yStripped[0] == '-') ? yStripped+1 : yStripped;
    if(strlen(xAbs) == 1 && strlen(yAbs) == 1) {
        char* product = multiplySingleDigits(xAbs, yAbs);
        if (((xStripped[0] == '-' && strcmp(xStripped, "0") != 0) ^ (yStripped[0] == '-' && strcmp(yStripped, "0") != 0)) && strcmp(product, "0") != 0) {
            char* res2 = (char*) malloc(strlen(product) + 2);
            res2[0] = '-';
            strcpy(res2 + 1, product);
            free(product);
            free(xStripped);
            free(yStripped);
            return res2;
        }
        free(xStripped);
        free(yStripped);
        return product;
    }
    
    int resultSign = 1;
    if(((xStripped[0] == '-' && strcmp(xStripped, "0") != 0) ^ (yStripped[0] == '-' && strcmp(yStripped, "0") != 0)))
        resultSign = -1;
    
    char* absX = (xStripped[0] == '-') ? strdup(xStripped+1) : strdup(xStripped);
    char* absY = (yStripped[0] == '-') ? strdup(yStripped+1) : strdup(yStripped);
    free(xStripped);
    free(yStripped);
    
    int lenX = strlen(absX), lenY = strlen(absY);
    int n = (lenX > lenY) ? lenX : lenY;
    while(n % 3 != 0)
        n++;
    char* X = (char*) malloc(n + 1);
    char* Y = (char*) malloc(n + 1);
    memset(X, '0', n - lenX);
    strcpy(X + (n - lenX), absX);
    X[n] = '\0';
    memset(Y, '0', n - lenY);
    strcpy(Y + (n - lenY), absY);
    Y[n] = '\0';
    free(absX);
    free(absY);
    
    int m = n / 3;
    char* X2 = strndup(X, m);
    char* X1 = strndup(X + m, m);
    char* X0 = strdup(X + 2 * m);
    char* Y2 = strndup(Y, m);
    char* Y1 = strndup(Y + m, m);
    char* Y0 = strdup(Y + 2 * m);
    free(X);
    free(Y);
    
    char* v0 = toom3(X0, Y0);
    
    char* temp1 = bigAddSigned(X2, X1);
    char* P1 = bigAddSigned(temp1, X0);
    free(temp1);
    char* temp2 = bigAddSigned(Y2, Y1);
    char* Q1 = bigAddSigned(temp2, Y0);
    free(temp2);
    char* v1 = toom3(P1, Q1);
    free(P1);
    free(Q1);
    
    char* temp3 = bigSubtractSigned(X2, X1);
    char* Pm1 = bigAddSigned(temp3, X0);
    free(temp3);
    char* temp4 = bigSubtractSigned(Y2, Y1);
    char* Qm1 = bigAddSigned(temp4, Y0);
    free(temp4);
    char* vNeg1 = toom3(Pm1, Qm1);
    free(Pm1);
    free(Qm1);
    
    char* fourX2 = bigMultiplyByInt(X2, 4);
    char* twoX1 = bigMultiplyByInt(X1, 2);
    char* temp5 = bigAddSigned(fourX2, twoX1);
    free(fourX2);
    free(twoX1);
    char* P2 = bigAddSigned(temp5, X0);
    free(temp5);
    
    char* fourY2 = bigMultiplyByInt(Y2, 4);
    char* twoY1 = bigMultiplyByInt(Y1, 2);
    char* temp6 = bigAddSigned(fourY2, twoY1);
    free(fourY2);
    free(twoY1);
    char* Q2 = bigAddSigned(temp6, Y0);
    free(temp6);
    
    char* v2 = toom3(P2, Q2);
    free(P2);
    free(Q2);
    
    char* vInf = toom3(X2, Y2);
    
    char* r0 = strdup(v0);
    char* r4 = strdup(vInf);
    
    char* sum_v1_vNeg1 = bigAddSigned(v1, vNeg1);
    char* sum_r0_r4 = bigAddSigned(r0, r4);
    char* two_r0_r4 = bigMultiplyByInt(sum_r0_r4, 2);
    char* numerator_r2 = bigSubtractSigned(sum_v1_vNeg1, two_r0_r4);
    free(sum_v1_vNeg1);
    free(sum_r0_r4);
    free(two_r0_r4);
    char* r2 = bigDivideByInt(numerator_r2, 2);
    free(numerator_r2);
    
    char* diff_v1_vNeg1 = bigSubtractSigned(v1, vNeg1);
    char* r1_plus_r3 = bigDivideByInt(diff_v1_vNeg1, 2);
    free(diff_v1_vNeg1);

    char* four_r2 = bigMultiplyByInt(r2, 4);
    char* sixteen_r4 = bigMultiplyByInt(r4, 16);
    char* temp7 = bigAddSigned(r0, four_r2);
    char* temp8 = bigAddSigned(temp7, sixteen_r4);
    free(four_r2);
    free(sixteen_r4);
    free(temp7);
    char* diff_v2 = bigSubtractSigned(v2, temp8);
    char* r1_plus_4r3 = bigDivideByInt(diff_v2, 2);
    free(diff_v2);
    
    char* temp9 = bigSubtractSigned(r1_plus_4r3, r1_plus_r3);
    char* r3 = bigDivideByInt(temp9, 3);
    free(temp9);
    char* r1 = bigSubtractSigned(r1_plus_r3, r3);
    
    char* r4_shift = multiplyByPowerOf10(r4, 4 * m);
    char* r3_shift = multiplyByPowerOf10(r3, 3 * m);
    char* r2_shift = multiplyByPowerOf10(r2, 2 * m);
    char* r1_shift = multiplyByPowerOf10(r1, m);
    char* tempSum1 = bigAddSigned(r4_shift, r3_shift);
    char* tempSum2 = bigAddSigned(r2_shift, r1_shift);
    char* tempSum3 = bigAddSigned(tempSum1, tempSum2);
    char* result = bigAddSigned(tempSum3, r0);
    
    free(X2); free(X1); free(X0);
    free(Y2); free(Y1); free(Y0);
    free(v0); free(v1); free(vNeg1); free(v2); free(vInf);
    free(r0); free(r4); free(r2);
    free(r1_plus_r3); free(r1_plus_4r3);
    free(r3); free(r1);
    free(r4_shift); free(r3_shift); free(r2_shift); free(r1_shift);
    free(tempSum1); free(tempSum2); free(tempSum3);
    
    if(resultSign < 0 && result[0] != '-') {
        char* res2 = (char*) malloc(strlen(result) + 2);
        res2[0] = '-';
        strcpy(res2 + 1, result);
        free(result);
        result = res2;
    }
    
    char* finalResult = stripLeadingZerosSigned(result);
    free(result);
    return finalResult;
}

int main() {
    char a[11000], b[11000];
    scanf("%s", a);
    scanf("%s", b);
    
    char* result = toom3(a, b);
    printf("%s", result);
    free(result);
    return 0;
}
