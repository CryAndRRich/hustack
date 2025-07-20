int minSwaps(int* data, int dataSize) {
    int ones = 0;
    for (int i = 0; i < dataSize; i++) {
        ones += data[i];
    }
    if (ones <= 1) return 0;

    int currentOnes = 0;
    for (int i = 0; i < ones; i++) {
        currentOnes += data[i];
    }

    int maxOnes = currentOnes;
    for (int i = ones; i < dataSize; i++) {
        currentOnes += data[i] - data[i - ones];
        if (currentOnes > maxOnes) maxOnes = currentOnes;
    }

    return ones - maxOnes;
}
