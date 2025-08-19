## Description
You are given an integer array $power$ where $power[i]$ is the power of the $i^{th}$ monster.

You start with $0$ mana points, and each day you increase your mana points by gain where gain initially is equal to $1$.

Each day, after gaining gain mana, you can defeat a monster if your mana points are greater than or equal to the power of that monster. When you defeat a monster:
- your mana points will be reset to $0$, and
- the value of gain increases by $1$.

Return the minimum number of days needed to defeat all the monsters.

## Example
### Example 1:
**Input**: $power = [3,1,4]$  
**Output**: $4$  
**Explanation**: The optimal way to beat all the monsters is
- Day $1$: Gain $1$ mana point to get a total of $1$ mana point. Spend all mana points to kill the $2^{nd}$ monster.
- Day $2$: Gain $2$ mana points to get a total of $2$ mana points.
- Day $3$: Gain $2$ mana points to get a total of $4$ mana points. Spend all mana points to kill the $3^{rd}$ monster.
- Day $4$: Gain $3$ mana points to get a total of $3$ mana points. Spend all mana points to kill the $1^{st}$ monster.

It can be proven that 4 is the minimum number of days needed. 

### Example 2:
**Input**: $power = [1,1,4]$  
**Output**: $4$  
**Explanation**: The optimal way to beat all the monsters is
- Day $1$: Gain $1$ mana point to get a total of $1$ mana point. Spend all mana points to kill the $1^{st}$ monster.
- Day $2$: Gain $2$ mana points to get a total of $2$ mana points. Spend all mana points to kill the $2^{nd}$ monster.
- Day $3$: Gain $3$ mana points to get a total of $3$ mana points.
- Day $4$: Gain $3$ mana points to get a total of $6$ mana points. Spend all mana points to kill the $3^{rd}$ monster.

It can be proven that 4 is the minimum number of days needed. 

### Example 3:
**Input**: $power = [1,2,4,9]$  
**Output**: $6$  
**Explanation**: The optimal way to beat all the monsters is
- Day $1$: Gain $1$ mana point to get a total of $1$ mana point. Spend all mana points to kill the $1^{st}$ monster.
- Day $2$: Gain $2$ mana points to get a total of $2$ mana points. Spend all mana points to kill the $2^{nd}$ monster.
- Day $3$: Gain $3$ mana points to get a total of $3$ mana points.
- Day $4$: Gain $3$ mana points to get a total of $6$ mana points.
- Day $5$: Gain $3$ mana points to get a total of $9$ mana points. Spend all mana points to kill the $4^{th}$ monster.
- Day $6$: Gain $4$ mana points to get a total of $4$ mana points. Spend all mana points to kill the $3^{rd}$ monster.

It can be proven that 6 is the minimum number of days needed.
 
## Constraints:
- $1 \leq power.length \leq 17$
- $1 \leq power[i] \leq 10^9$