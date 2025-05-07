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

## Result:
<table style="text-align: center;">
    <thead>
        <tr>
            <th style="text-align: center;">File</th>
            <th colspan="2" style="text-align: center;">Test 1</th>
            <th colspan="2" style="text-align: center;">Test 2</th>
            <th colspan="2" style="text-align: center;">Test 3</th>
            <th colspan="2" style="text-align: center;">Test 4</th>
            <th colspan="2" style="text-align: center;">Test 5</th>
        </tr>
        <tr>
            <th></th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>recursive.py</td>
            <td>41</td>
            <td>3.191</td>
            <td>41</td>
            <td>3.082</td>
            <td>42</td>
            <td>3.191</td>
            <td>47</td>
            <td>3.156</td>
            <td>41</td>
            <td>3.133</td>
        </tr>
        <tr>
            <td>stack.py</td>
            <td>41</td>
            <td>3.156</td>
            <td>42</td>
            <td>3.176</td>
            <td>41</td>
            <td>3.152</td>
            <td>54</td>
            <td>3.141</td>
            <td>42</td>
            <td>3.191</td>
        </tr>
        <tr>
            <td>stack.c</td>
            <td><strong>6</strong></td>
            <td><strong>0.828</strong></td>
            <td><strong>6</strong></td>
            <td>0.836</td>
            <td><strong>6</strong></td>
            <td><strong>0.719</strong></td>
            <td><strong>6</strong></td>
            <td><strong>0.844</strong></td>
            <td>6</td>
            <td><strong>0.852</strong></td>
        </tr>
        <tr>
            <td>stack.cpp</td>
            <td><strong>6</strong></td>
            <td>0.855</td>
            <td><strong>6</strong></td>
            <td><strong>0.832</strong></td>
            <td>7</td>
            <td>0.891</td>
            <td><strong>6</strong></td>
            <td>0.914</td>
            <td><strong>5</strong></td>
            <td>0.898</td>
        </tr>
    </tbody>
</table>
