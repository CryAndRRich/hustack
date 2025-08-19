class Matrix3D {
private:
    int n;
    vector<vector<vector<int>>> matrix;
    vector<int> count;

public:
    Matrix3D(int n) {
        this->n = n;
        matrix.assign(n, vector<vector<int>>(n, vector<int>(n, 0)));
        count.assign(n, 0);
    }

    void setCell(int x, int y, int z) {
        if (matrix[x][y][z] == 0) {
            matrix[x][y][z] = 1;
            count[x]++;
        }
    }

    void unsetCell(int x, int y, int z) {
        if (matrix[x][y][z] == 1) {
            matrix[x][y][z] = 0;
            count[x]--;
        }
    }

    int largestMatrix() {
        int maxCount = *max_element(count.begin(), count.end());
        for (int i = n - 1; i >= 0; i--) {
            if (count[i] == maxCount) return i;
        }
        return -1;
    }
};
