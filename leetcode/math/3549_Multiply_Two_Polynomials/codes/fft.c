#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

void fft(complex double* a, int n, int invert) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            complex double temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        complex double wlen = cos(ang) + sin(ang) * I;
        for (int i = 0; i < n; i += len) {
            complex double w = 1;
            for (int j = 0; j < len / 2; ++j) {
                complex double u = a[i + j];
                complex double v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; ++i)
            a[i] /= n;
    }
}

long long* multiply(int* poly1, int poly1Size, int* poly2, int poly2Size, int* returnSize) {
    int n = 1;
    while (n < poly1Size + poly2Size)
        n <<= 1;

    complex double* fa = (complex double*)calloc(n, sizeof(complex double));
    complex double* fb = (complex double*)calloc(n, sizeof(complex double));
    for (int i = 0; i < poly1Size; i++) fa[i] = poly1[i];
    for (int i = 0; i < poly2Size; i++) fb[i] = poly2[i];

    fft(fa, n, 0);
    fft(fb, n, 0);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, n, 1);

    long long* result = (long long*)malloc(n * sizeof(long long));
    for (int i = 0; i < n; i++)
        result[i] = llround(creal(fa[i]));

    *returnSize = poly1Size + poly2Size - 1;
    free(fa);
    free(fb);
    return result;
}
