#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAXN 1001
#define MAXQ 100000

typedef struct {
    int sum, steps;
} Node;

int minNumberOfPrimes(int n, int m) {
    bool isPrime[8000] = {false};
    int primes[1000], pCount = 0;
    for (int i = 2; i < 8000 && pCount < m; i++) {
        isPrime[i] = true;
    }
    for (int i = 2; i * i < 8000; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j < 8000; j += i)
                isPrime[j] = false;
        }
    }
    for (int i = 2; i < 8000 && pCount < m; i++) {
        if (isPrime[i]) primes[pCount++] = i;
    }

    bool visited[MAXN] = {false};
    Node queue[MAXQ];
    int front = 0, rear = 0;

    queue[rear++] = (Node){0, 0};
    visited[0] = true;

    while (front < rear) {
        Node current = queue[front++];
        if (current.sum == n) return current.steps;

        for (int i = 0; i < pCount; ++i) {
            int next = current.sum + primes[i];
            if (next <= n && !visited[next]) {
                visited[next] = true;
                queue[rear++] = (Node){next, current.steps + 1};
            }
        }
    }

    return -1;
}
