#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define M_PI 3.14159265358979323846
typedef double complex cd;

void fft(cd *a, int n, int invert) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            cd tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        cd wlen = cos(ang) + I * sin(ang);
        for (int i = 0; i < n; i += len) {
            cd w = 1;
            for (int j = 0; j < len/2; j++) {
                cd u = a[i+j];
                cd v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; i++)
            a[i] /= n;
    }
}

char* fftMultiply(const char* s1, const char* s2) {
    if ((strlen(s1) == 1 && s1[0] == '0') || (strlen(s2) == 1 && s2[0] == '0')) {
        char* res_zero = malloc(2);
        if (!res_zero) exit(1);
        res_zero[0] = '0';
        res_zero[1] = '\0';
        return res_zero;
    }
    
    int n1 = strlen(s1), n2 = strlen(s2);
    int n = 1;
    while (n < n1 + n2)
        n <<= 1;
        
    cd *fa = (cd*) calloc(n, sizeof(cd));
    cd *fb = (cd*) calloc(n, sizeof(cd));

    for (int i = 0; i < n1; i++)
        fa[i] = s1[n1 - 1 - i] - '0';
    for (int i = 0; i < n2; i++)
        fb[i] = s2[n2 - 1 - i] - '0';
    
    fft(fa, n, 0);
    fft(fb, n, 0);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, n, 1);
    
    int *res = (int*) calloc(n, sizeof(int));

    for (int i = 0; i < n; i++)
        res[i] = (int)round(creal(fa[i]));
    
    for (int i = 0; i < n; i++) {
        if (res[i] >= 10) {
            if (i + 1 == n) {
                res = realloc(res, sizeof(int) * (n + 1));

                res[n] = 0;
                n++;
            }
            res[i+1] += res[i] / 10;
            res[i] %= 10;
        }
    }
    
    int len = n;
    while (len > 1 && res[len-1] == 0)
        len--;
    
    char *result = (char*) malloc(len + 1);

    for (int i = 0; i < len; i++)
        result[i] = res[len - 1 - i] + '0';
    result[len] = '\0';
    
    free(res);
    free(fa);
    free(fb);
    return result;
}

int main() {
    char a[11000], b[11000];
    scanf("%s", a);
    scanf("%s", b);
    
    char* result = fftMultiply(a, b);
    printf("%s\n", result);
    free(result);
    return 0;
}
