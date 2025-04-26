#include <iostream>
#include <vector>
using namespace std;

int main(){
    int n, m;
    cin >> n >> m;
    
    vector<vector<int>> A(n + 1, vector<int>(m + 1, 0));
    vector<vector<int>> S(n + 1, vector<int>(m + 1, 0));
    
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= m; j++){
            cin >> A[i][j];
        }
    }
    
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= m; j++){
            S[i][j] = A[i][j] + S[i - 1][j] + S[i][j - 1] - S[i - 1][j - 1];
        }
    }
    
    int maxArea = 0;
    
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= m; j++){
            if (S[i][j] == i * j) {
                int area = i * j;
                if (area > maxArea)
                    maxArea = area;
            }
        }
    }
    
    cout << maxArea;
    return 0;
}
