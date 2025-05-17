## Description
Given a string containing only characters (, ), [, ] {, }. Write a program that check whether the string is correct in expression

**Input:** One line contains the string (the length of the string is less than or equal to $10^6$)

**Output:** Write $1$ if the sequence is correct, and write $0$, otherwise

## Approach: Stack
Traverse each character in the string one by one:
- If the character is an opening bracket (, [, or {, push it onto the stack
- If the character is a closing bracket ), ], or }:
    - If the stack is empty → the expression is invalid (return $0$)
    - Otherwise, pop the top element from the stack and check:
        - If it does not match the correct opening bracket for the current closing bracket, invalid (return $0$)

After processing all characters:
- If the stack is empty, the expression is valid (return $1$)
- If the stack still contains any items, invalid (return $0$)

**Time Complexity:** $O(n)$

**Space Complexity:** $O(n)$
