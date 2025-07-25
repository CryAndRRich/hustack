## Description
Given an array of unique strings $words$, return all the word squares you can build from $words$. The same word from $words$ can be used multiple times. You can return the answer in any order.

A sequence of strings forms a valid word square if the $k^{th}$ row and column read the same string, where $0 \leq k < max(numRows, numColumns)$

For example, the word sequence $["ball","area","lead","lady"]$ forms a word square because each word reads the same both horizontally and vertically.
 
## Example
### Example 1:
**Input**: $words = ["area","lead","wall","lady","ball"]$  
**Output**: $[["ball","area","lead","lady"],["wall","area","lead","lady"]]$  
**Explanation**: The output consists of two word squares. The order of output does not matter (just the order of words in each word square matters)

### Example 2:
**Input**: $words = ["abat","baba","atan","atal"]$  
**Output**: $[["baba","abat","baba","atal"],["baba","abat","baba","atan"]]$  
**Explanation**: The output consists of two word squares. The order of output does not matter (just the order of words in each word square matters)
 
## Constraints:
- $1 \leq words.length \leq 1000$
- $1 \leq words[i].length \leq 4$
- All $words[i]$ have the same length
- $words[i]$ consists of only lowercase English letters
- All $words[i]$ are unique