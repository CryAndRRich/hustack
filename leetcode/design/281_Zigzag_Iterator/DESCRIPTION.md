## Description
Given two vectors of integers $v_1$ and $v_2$, implement an iterator to return their elements alternately.

Implement the $ZigzagIterator$ class:
- $ZigzagIterator(List<int> v1, List<int> v2)$ initializes the object with the two vectors $v_1$ and $v_2$.
- $boolean hasNext()$ returns true if the iterator still has elements, and false otherwise.
- $int next()$ returns the current element of the iterator and moves the iterator to the next element.
 
## Example
### Example 1:
**Input**: $v_1 = [1,2], v_2 = [3,4,5,6]$  
**Output**: $[1,3,2,4,5,6]$  
**Explanation**: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: $[1,3,2,4,5,6]$

### Example 2:
**Input**: $v_1 = [1], v_2 = []$  
**Output**: $[1]$

### Example 3:
**Input**: $v_1 = [], v_2 = [1]$  
**Output**: $[1]$
 
## Constraints:
- $0 <= v_1.length, v_2.length <= 1000$
- $1 <= v_1.length + v_2.length <= 2000$
- $-2^{31} <= v_1[i], v_2[i] <= 2^{31} - 1$