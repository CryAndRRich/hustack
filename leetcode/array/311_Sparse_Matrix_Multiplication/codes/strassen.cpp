#include <vector>
using namespace std;

class Solution {
public:
    vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
        int n = A.size();
        return strassen(A, B, n);
    }

private:
    vector<vector<int>> add(const vector<vector<int>>& A, const vector<vector<int>>& B) {
        int n = A.size();
        vector<vector<int>> C(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    vector<vector<int>> subtract(const vector<vector<int>>& A, const vector<vector<int>>& B) {
        int n = A.size();
        vector<vector<int>> C(n, vector<int>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    vector<vector<int>> strassen(const vector<vector<int>>& A, const vector<vector<int>>& B, int n) {
        if (n == 1) return {{A[0][0] * B[0][0]}};

        int mid = n / 2;
        vector<vector<int>> A11(mid, vector<int>(mid)), A12(mid, vector<int>(mid)),
                           A21(mid, vector<int>(mid)), A22(mid, vector<int>(mid)),
                           B11(mid, vector<int>(mid)), B12(mid, vector<int>(mid)),
                           B21(mid, vector<int>(mid)), B22(mid, vector<int>(mid));

        for (int i = 0; i < mid; ++i) {
            for (int j = 0; j < mid; ++j) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + mid];
                A21[i][j] = A[i + mid][j];
                A22[i][j] = A[i + mid][j + mid];
                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j + mid];
                B21[i][j] = B[i + mid][j];
                B22[i][j] = B[i + mid][j + mid];
            }
        }

        auto M1 = strassen(add(A11, A22), add(B11, B22), mid);
        auto M2 = strassen(add(A21, A22), B11, mid);
        auto M3 = strassen(A11, subtract(B12, B22), mid);
        auto M4 = strassen(A22, subtract(B21, B11), mid);
        auto M5 = strassen(add(A11, A12), B22, mid);
        auto M6 = strassen(subtract(A21, A11), add(B11, B12), mid);
        auto M7 = strassen(subtract(A12, A22), add(B21, B22), mid);

        vector<vector<int>> C(n, vector<int>(n));
        for (int i = 0; i < mid; ++i) {
            for (int j = 0; j < mid; ++j) {
                C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                C[i][j + mid] = M3[i][j] + M5[i][j];
                C[i + mid][j] = M2[i][j] + M4[i][j];
                C[i + mid][j + mid] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }
        }

        return C;
    }
};