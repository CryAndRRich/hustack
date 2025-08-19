## Description
You are given an integer n representing the number of playing cards you have. A house of cards meets the following conditions:
- A house of cards consists of one or more rows of triangles and horizontal cards.
- Triangles are created by leaning two cards against each other.
- One card must be placed horizontally between all adjacent triangles in a row.
- Any triangle on a row higher than the first must be placed on a horizontal card from the previous row.
- Each triangle is placed in the leftmost available spot in the row.

Return the number of distinct house of cards you can build using all $n$ cards. Two houses of cards are considered distinct if there exists a row where the two houses contain a different number of cards.

## Example
### Example 1:
**Input**: $n = 16$  
**Output**: $2$ 

### Example 2:
**Input**: $n = 2$  
**Output**: $1$

### Example 3:
**Input**: $n = 4$  
**Output**: $0$  

## Constraints:
- $1 \leq n \leq 500$