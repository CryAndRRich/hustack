import numpy as np
from typing import List

class Solution:
    def multiply(self, poly1: List[int], poly2: List[int]) -> List[int]:
        n = len(poly1) + len(poly2) - 1
        size = 1 << (n - 1).bit_length()  
        
        fft1 = np.fft.fft(poly1, size)
        fft2 = np.fft.fft(poly2, size)
        fft_result = fft1 * fft2
        
        result = np.fft.ifft(fft_result).real
        result = np.round(result).astype(int).tolist()
        return result[:n]
