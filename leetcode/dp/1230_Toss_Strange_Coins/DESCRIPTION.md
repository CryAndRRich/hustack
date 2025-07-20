## Description
You have some coins.  The $i^{th}$ coin has a probability $prob[i]$ of facing heads when tossed.
Return the probability that the number of coins facing heads equals $target$ if you toss every coin exactly once.

## Example
### Example 1:
**Input**: $prob = [0.4], target = 1$  
**Output**: $0.40000$

### Example 2:
**Input**: $prob = [0.5,0.5,0.5,0.5,0.5], target = 0$  
**Output**: $0.03125$
 
## Constraints:
- $1 \leq prob.length \leq 1000$
- $0 \leq prob[i] \leq 1$
- $0 \leq target \leq prob.length$
- Answers will be accepted as correct if they are within $10^-5$ of the correct answer