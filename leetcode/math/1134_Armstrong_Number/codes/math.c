#include <stdbool.h>
#include <math.h>

bool isArmstrong(int n) {
    int original = n, sum = 0, k = 0, temp = n;

    while (temp > 0) {
        temp /= 10;
        k++;
    }

    temp = n;
    while (temp > 0) {
        int digit = temp % 10;
        sum += pow(digit, k);
        temp /= 10;
    }

    return sum == original;
}
