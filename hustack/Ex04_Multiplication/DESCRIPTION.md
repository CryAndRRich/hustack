## Description
Given two very large positive integers $a$ and $b$ (the number of digits can be up to $10000$). Calculate and print the product of these two numbers.

**Input**: 
- Line $1$: contains the number $a$
- Line $2$: contains the number $b$

**Output**:
The result $a.b$

## Handling Large Inputs as Strings

- **Input Size:** The input numbers can be as long as $10,000$ digits. This is far beyond the capacity of any built-in numeric type. Therefore, the program reads the numbers as strings and implements arithmetic manually.

- **Why Strings?** Using strings allows us to process each digit individually. We can simulate the traditional pencil-and-paper method for addition, subtraction, and multiplication, which makes it possible to work with numbers of arbitrary length.

### String-based Arithmetic Operations:

#### Addition:
- **Process:** The function adds two numbers by iterating from the end (least significant digit) of each string towards the beginning.

- **Steps:**  
  - It initializes a carry variable to handle sums greater than $9$.
  - For each pair of digits (or a digit and $0$ if one number is shorter), it computes the sum, adds the carry, and determines the new carry.
  - The resulting digit is stored in a result string.
  - Finally, the result string (which might have extra leading zeros) is trimmed if necessary.

- **Note:**  
  This process mimics the manual addition method we use on paper.

#### Subtraction:
- **Process:**  
  The function subtracts one large number from another by processing the digits from right to left.

- **Steps:**  
  - It handles borrowing if the digit of the minuend (the number being subtracted from) is smaller than the corresponding digit of the subtrahend.
  - It subtracts each pair of digits, adjusts for any borrow, and stores the result in a new string.
  - After the operation, it strips any leading zeros that might have resulted.

- **Note:**  
  Like addition, this function replicates the manual subtraction method.

#### Multiplication by a Power of 10:
- **Process:**  
  Multiplying a number by $10^m$ simply involves appending $m$ zeros to the end of its string representation.

- **Steps:**  
  - The function creates a new string that copies the original number.
  - It then appends the required number of '$0$' characters at the end.

- **Use Case:**  
  This operation is essential for combining the results of the Karatsuba algorithm, where partial results need to be shifted appropriately (e.g., $Z_2 \times 10^{2m}$).

#### Single-digit Multiplication:
- **Process:**  
  When the algorithm reaches a base case (both numbers are single-digit), it directly multiplies them.

- **Steps:**  
  - It converts the characters to their integer values, multiplies them, and then converts the result back to a string.

## Approach 1: Karatsuba Multiplication
We implement the **Karatsuba algorithm** for multiplying large numbers represented as strings. The Karatsuba algorithm is a **divide and conquer** approach that significantly reduces the number of single-digit multiplications compared to the traditional **grade-school multiplication method**.

### Karatsuba Multiplication Algorithm
- **Overview:** The Karatsuba algorithm is a divide-and-conquer method that reduces the multiplication of two $n$-digit numbers from $O(n^2)$ to approximately $O(n^{1.585})$. It does this by splitting the numbers into halves and recursively calculating three products instead of four.

- **Steps in the Code:**  
  **Preprocessing:**  
    - Both numbers are stripped of leading zeros.
    - They are padded with zeros if necessary to ensure they have the same even length.
  
  **Splitting:**  
    - Each number is split into a high part (first half) and a low part (second half).
  
  **Recursive Multiplications:**  
    - $Z_2 = \text{karatsuba}(X_{high}, Y_{high})$
    - $Z_0 = \text{karatsuba}(X_{low}, Y_{low})$
    - $Z_1 = \text{karatsuba}(X_{high} + X_{low}, Y_{high} + Y_{low}) - Z_2 - Z_0$
  
  **Combining the Results:**  
    - The final result is computed as:  
    $$\text{Result} = Z_2 \times 10^{2m} + Z_1 \times 10^{m} + Z_0$$

- **Memory Management:**  
  - Each function dynamically allocates memory for intermediate results.
  - Care is taken to free this memory after its use to avoid memory leaks.

### Time Complexity:
- The Karatsuba algorithm splits each number into two halves.
- For two $n$-digit numbers, it performs three recursive multiplications on numbers of roughly size $n/2$.

Therefore, the overall time complexity is $O(n^{\log _23})≈O(n^{1.585})$
### Space Complexity:
The overall auxiliary space complexity is $O(n)$

## Approach 2: Toom–Cook Multiplication
The Toom–Cook algorithm is a divide-and-conquer method that splits each large number into three smaller parts (for Toom-3) and evaluates a polynomial at several points. By doing so, it reduces the multiplication of two 
$n$-digit numbers into five multiplications of numbers with roughly $n/3$ digits (plus additional linear-time operations)
### Toom-3 Algorithm
- **Overview:** The Toom–Cook algorithm is a divide-and-conquer method that splits each large number into three smaller parts (for Toom-3) and evaluates a polynomial at several points
- **Steps in the Code:**  
  **Preprocessing:**  
    - **Strip and Sign Handling:** Both input numbers are stripped of leading zeros. The sign is processed separately using helper functions.
    - **Padding:** The numbers are padded with zeros so that their lengths become divisible by 3. This ensures that they can be evenly split into three parts.

  **Splitting:**  
    - Each number is divided into three parts:
      - For the first number: $X_2$ (high), $X_1$ (middle), and $X_0$ (low).
      - For the second number: $Y_2$, $Y_1$, and $Y_0$.

  **Evaluation:**  
    - **Polynomial Evaluation:** The algorithm evaluates the “polynomials” represented by the three parts at several points:
      - At $0$: $v_0 = \text{toom3}(X_0, Y_0)$
      - At $1$: $v_1 = \text{toom3}(X_2 + X_1 + X_0,\; Y_2 + Y_1 + Y_0)$
      - At $-1$: $v_{-1} = \text{toom3}(X_2 - X_1 + X_0,\; Y_2 - Y_1 + Y_0)$
      - At $2$ (or by computing a weighted sum): $v_2 = \text{toom3}(4X_2 + 2X_1 + X_0,\; 4Y_2 + 2Y_1 + Y_0)$
      - At $\infty$: $v_{\infty} = \text{toom3}(X_2, Y_2)$

  **Interpolation:**  
    - Using the values $v_0$, $v_1$, $v_{-1}$, $v_2$, and $v_{\infty}$, the algorithm interpolates to recover the coefficients (denoted as $r_0, r_1, r_2, r_3, r_4$) of the product polynomial.
    - This step involves several additions, subtractions, multiplications by small integers, and divisions by small integers—all done in linear time relative to the length of the numbers involved.

  **Recomposition:**  
    - Each coefficient is then shifted appropriately (i.e., multiplied by powers of $10$) to position it correctly in the final number.
    - The shifted values are added together to form the final product.

  **Memory Management:**  
    - As with the Karatsuba implementation, the code makes extensive use of dynamic memory allocation for storing intermediate strings.
    - Each allocated block is freed once it is no longer needed, ensuring that the program does not leak memory.

### Time Complexity:
- The Karatsuba algorithm splits each number into three parts.
- For two $n$-digit numbers, it performs five recursive multiplications on numbers of roughly size $n/3$.

Therefore, the overall time complexity is $O(n^{\log _35})≈O(n^{1.465})$
### Space Complexity:
The overall auxiliary space complexity is $O(n)$

## Approach 3: Fast Fourier Transform - Schönhage–Strassen Algorithm
### FFT Algorithm
- **Overview:** The FFT-based multiplication algorithm multiplies two large numbers by representing them as polynomials, where each digit is treated as a coefficient. It leverages the Fast Fourier Transform (FFT) to efficiently compute the convolution of these polynomials in $O(n \log n)$ time. After the convolution, the algorithm rounds the resulting coefficients and performs carry propagation to produce the final product.

- **Steps in the Code:**
  **Preprocessing:**  
    - **Input Conversion and Padding:** The input numbers (provided as strings) are converted into arrays of digits in reverse order. These arrays are then padded with zeros to the next power of two, ensuring the FFT operates correctly.

  **FFT Computation:**  
    - **Forward FFT:** The FFT is applied to both arrays of digits, transforming them from the time domain into the frequency domain. This step has a time complexity of $O(n \log n)$.
    - **Pointwise Multiplication:** The corresponding elements of the two transformed arrays are multiplied together. This operation computes the convolution of the two digit arrays in the frequency domain.

  **Inverse FFT and Postprocessing:**  
    - **Inverse FFT:** An inverse FFT is performed on the product array to transform it back into the time domain. The resulting array contains real numbers that approximate the coefficients of the product polynomial.
    - **Rounding and Carry Propagation:** Each coefficient is rounded to the nearest integer. The algorithm then processes these coefficients to handle carries, ensuring each digit is within the range $[0, 9]$.
    - **Result Construction:** Finally, the digits are recombined in the correct order to form the resulting product as a string.

  **Memory Management:**  
  - The algorithm uses dynamic memory allocation for arrays of complex numbers and the result array.
  - All allocated memory is freed after use to prevent memory leaks.

### Time Complexity:
The FFT operates in $O(n \log n)$ time, where $n$ is the length of the padded digit arrays (a power of two).
### Space Complexity:
The overall auxiliary space complexity is $O(n)$
