#include <stdio.h>
#include <string.h>

static int dp[101][101][101];
static _Bool desired[101][101][101];

int main(){
    int T;
    scanf("%d", &T);
    while(T--){
        int L, W, H, N;
        scanf("%d %d %d", &L, &W, &H);
        scanf("%d", &N);
        for(int i = 1; i <= L; i++)
            for(int j = 1; j <= W; j++)
                for(int k = 1; k <= H; k++)
                    desired[i][j][k] = 0;

        while(N--){
            int a, b, c;
            scanf("%d %d %d", &a, &b, &c);
            int perms[6][3] = {
                {a, b, c}, {a, c, b},
                {b, a, c}, {b, c, a},
                {c, a, b}, {c, b, a}
            };
            for(int p = 0; p < 6; p++){
                int x = perms[p][0];
                int y = perms[p][1];
                int z = perms[p][2];
                if (x <= L && y <= W && z <= H)
                    desired[x][y][z] = 1;
            }
        }

        for(int i = 1; i <= L; i++){
            for(int j = 1; j <= W; j++){
                for(int k = 1; k <= H; k++){
                    if(desired[i][j][k]){
                        dp[i][j][k] = 0;
                    } else {
                        int best = i * j * k;
                        for(int x = 1; x <= i / 2; x++)
                            best = fmin(best, dp[x][j][k] + dp[i - x][j][k]);
                        for(int y = 1; y <= j / 2; y++)
                            best = fmin(best, dp[i][y][k] + dp[i][j - y][k]);
                        for(int z = 1; z <= k / 2; z++)
                            best = fmin(best, dp[i][j][z] + dp[i][j][k - z]);
                        dp[i][j][k] = best;
                    }
                }
            }
        }

        printf("%d\n", dp[L][W][H]);
    }
    return 0;
}
