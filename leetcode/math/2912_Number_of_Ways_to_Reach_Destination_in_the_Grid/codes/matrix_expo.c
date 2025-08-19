#include <stdio.h>
#include <string.h>

#define MOD 1000000007

void multiplyMatrix(int m, long long A[4][4], long long B[4][4], long long C[4][4]) {
    long long temp[4][4] = {0};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < m; ++k)
                temp[i][j] = (temp[i][j] + A[i][k] * B[k][j]) % MOD;

    memcpy(C, temp, sizeof(temp));
}

void multiplyVector(int m, long long vec[4], long long mat[4][4], long long res[4]) {
    long long temp[4] = {0};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            temp[i] = (temp[i] + vec[j] * mat[i][j]) % MOD;

    memcpy(res, temp, sizeof(temp));
}

void matrixPower(int m, long long mat[4][4], int exp, long long res[4]) {
    long long base[4][4];
    memcpy(base, mat, sizeof(base));

    while (exp) {
        if (exp % 2) {
            multiplyVector(m, res, base, res);
        }
        multiplyMatrix(m, base, base, base);
        exp /= 2;
    }
}

int numberOfWays(int n, int m, int k, int* source, int sourceSize, int* dest, int destSize) {
    long long matrix[4][4] = {
        {0, 1, 1, 0},
        {n - 1, n - 2, 0, 1},
        {m - 1, 0, m - 2, 1},
        {0, m - 1, n - 1, m + n - 4}
    };

    long long res[4] = {0};
    if (source[0] == dest[0] && source[1] == dest[1])
        res[0] = 1;
    else if (source[0] == dest[0])
        res[2] = 1;
    else if (source[1] == dest[1])
        res[1] = 1;
    else
        res[3] = 1;

    matrixPower(4, matrix, k, res);

    return (int)res[0];
}
