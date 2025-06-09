#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX 10000
#define INF 1000000000

double temperature;
double coolingRate;
const double T_MIN = 1e-3;

int n;
int dist[MAX][MAX];
int bestTour[MAX], currTour[MAX];
int bestCost, currCost;

int calculateCost(int *tour) {
    int cost = 0;
    for (int i = 0; i < n; i++) {
        cost += dist[tour[i]][tour[(i + 1) % n]];
    }
    return cost;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void simulatedAnnealing() {
    for (int i = 0; i < n; i++) {
        currTour[i] = i;
    }
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        swap(&currTour[i], &currTour[j]);
    }
    currCost = calculateCost(currTour);
    bestCost = currCost;
    for (int i = 0; i < n; i++) {
        bestTour[i] = currTour[i];
    }

    int iter = 0;
    int noImprovementCount = 0;
    const int NO_IMPROVEMENT_THRESHOLD = 100000;
    const int MAX_ITER = 10000000;

    while (temperature > T_MIN && iter < MAX_ITER) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            swap(&currTour[i], &currTour[j]);
            int newCost = calculateCost(currTour);
            int delta = newCost - currCost;

            if (delta < 0 || (exp(-delta / temperature) > ((double)rand() / RAND_MAX))) {
                currCost = newCost;
                if (currCost < bestCost) {
                    bestCost = currCost;
                    for (int k = 0; k < n; k++) {
                        bestTour[k] = currTour[k];
                    }
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                swap(&currTour[i], &currTour[j]);
                noImprovementCount++;
            }
        }
        iter++;
        if (noImprovementCount > NO_IMPROVEMENT_THRESHOLD) {
            temperature *= 1.1;
            noImprovementCount = 0;
        } else {
            temperature *= coolingRate;
        }
    }
}

int main() {
    srand(time(NULL));
    scanf("%d", &n);
    
    int temp;
    int maxDist = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &temp);
            dist[i][j] = temp;
            if (temp > maxDist) {
                maxDist = temp;
            }
        }
    }
    
    double alpha = 10.01;
    temperature = alpha * maxDist;
    coolingRate = 0.99999;

    simulatedAnnealing();

    printf("%d\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d ", bestTour[i] + 1);
    }
    return 0;
}
