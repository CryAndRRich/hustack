#include <bits/stdc.++h>
using namespace std;

int dp[101][101][101];
bool desired[101][101][101];

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while(T--){
        int L, W, H;
        cin >> L >> W >> H;
        int N;
        cin >> N;
        for(int i = 1; i <= L; i++)
            for(int j = 1; j <= W; j++)
                for(int k = 1; k <= H; k++)
                    desired[i][j][k] = false;

        while(N--){
            int a, b, c;
            cin >> a >> b >> c;
            int dims[3] = {a,b,c};
            sort(dims, dims + 3);
            do{
                desired[dims[0]][dims[1]][dims[2]] = true;
            } while(next_permutation(dims, dims + 3));
        }

        for(int i = 1; i <= L; i++){
            for(int j = 1; j <= W; j++){
                for(int k = 1; k <= H; k++){
                    if(desired[i][j][k]){
                        dp[i][j][k] = 0;
                    } else {
                        int best = i * j * k;
                        for(int x = 1; x <= i / 2; x++)
                            best = min(best, dp[x][j][k] + dp[i - x][j][k]);
                        for(int y = 1; y <= j / 2; y++)
                            best = min(best, dp[i][y][k] + dp[i][j - y][k]);
                        for(int z = 1; z <= k / 2; z++)
                            best = min(best, dp[i][j][z] + dp[i][j][k - z]);
                        dp[i][j][k] = best;
                    }
                }
            }
        }

        cout << dp[L][W][H] << "\n";
    }
    return 0;
}
