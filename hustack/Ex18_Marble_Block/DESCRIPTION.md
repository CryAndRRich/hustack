## Description
You are given an initial marble block of size $L \times W \times H$ and a set of $N$ desired sub‐blocks, each with dimensions $(L_i, W_i, H_i)$. You may cut any block along one of its three axes into two smaller integer‐sized blocks. Cuts must go all the way through. After cutting, pieces cannot be recombined. You can rotate any sub‐block (i.e. use any permutation of its dimensions) to match a desired size. You may produce zero or more copies of each desired size. Any piece that does not match one of the desired dimensions is considered wasted. Find a sequence of cuts that minimizes the total wasted volume.

**Input:**
- **Line 1**: three integers $L, W, H$ ($1 \le L, W, H \le 100$)
- **Line 2**: integer $N$ ($1 \le N \le 50$)
- **Next $N$ lines**: three integers $L_i, W_i, H_i$ ($1 \le L_i \le L$, $1 \le W_i \le W$, $1 \le H_i \le H$)

**Output:** A single integer - the minimum total volume of wasted marble, or $0$ if you can partition with no waste

## Approach: DP
Use a 3D dynamic‐programming table $dp[i][j][k]$ - minimum waste for a block of size $i\times j\times k$. Precompute all $6$ rotations of each desired dimension and store in a set desired.
For each $(i,j,k)$:
- If $(i,j,k)\in \text{desired}$, then $dp[i][j][k] = 0$
- Otherwise initialize $dp[i][j][k] = ijk$ (waste entire block), then try every cut along each axis:
    - For $x=1\ldots\lfloor i/2\rfloor$: 
    $$dp[i][j][k]=min(dp[i][j][k],dp[x][j][k]+dp[i−x][j][k])$$
    - Similarly cut along $j$ and $k$.

After filling up to $(L,W,H)$, the answer is $dp[L][W][H]$

**Time Complexity**: $O(LWH(L+W+H))$

**Space Complexity**: $O(LWH)$