#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define MAX 10000
#define INF 1000000000

void candidateSolution(int n, int **dist, int *solution, long long *solCost, double alpha) {
    int curr = rand() % n;  
    int count = 0;
    long long cost = 0;

    int *visited = (int*)calloc(n, sizeof(int));
    visited[curr] = 1;
    solution[count] = curr;
    count++;

    while(count < n) {
        int numCandidates = 0;
        int candidates[n];
        double candDist[n]; 

        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                candidates[numCandidates] = i;
                candDist[numCandidates] = dist[curr][i];
                numCandidates++;
            }
        }
        if(numCandidates == 0) break;

        double minDist = candDist[0], maxDist = candDist[0];
        for (int j = 0; j < numCandidates; j++) {
            if(candDist[j] < minDist) minDist = candDist[j];
            if(candDist[j] > maxDist) maxDist = candDist[j];
        }
        double threshold = minDist + alpha * (maxDist - minDist);

        int candidateList[numCandidates];
        int candidateCount = 0;
        for (int j = 0; j < numCandidates; j++) {
            if(candDist[j] <= threshold) {
                candidateList[candidateCount++] = j;
            }
        }

        int chosenIndex = candidateList[rand() % candidateCount];
        int chosen = candidates[chosenIndex];

        solution[count] = chosen;
        cost += dist[curr][chosen];
        curr = chosen;
        visited[chosen] = 1;
        count++;
    }
    cost += dist[curr][solution[0]];
    *solCost = cost;
    free(visited);
}

int calculateCost(int n, int **dist, int *tour) {
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

void simulatedAnnealingHybrid(int n, int **dist, int *initTour, int *bestTour, int *bestCost) {
    int currTour[n];
    for (int i = 0; i < n; i++) {
        currTour[i] = initTour[i];
        bestTour[i] = initTour[i];
    }
    int currCost = calculateCost(n, dist, currTour);
    *bestCost = currCost;

    double temperature;
    double coolingRate = 0.99999;
    const double T_MIN = 1e-3;
    
    int maxDist = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(dist[i][j] > maxDist) {
                maxDist = dist[i][j];
            }
        }
    }
    temperature = 10.0 * maxDist; 

    int iter = 0;
    const int MAX_ITER = 10000000;
    int noImprovementCount = 0;
    const int NO_IMPROVEMENT_THRESHOLD = 100000;

    while (temperature > T_MIN && iter < MAX_ITER) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            swap(&currTour[i], &currTour[j]);
            int newCost = calculateCost(n, dist, currTour);
            int delta = newCost - currCost;

            if (delta < 0 || (exp(-delta / temperature) > ((double)rand() / RAND_MAX))) {
                currCost = newCost;
                if (currCost < *bestCost) {
                    *bestCost = currCost;
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
int main(){
    srand(time(NULL));
    int n;
    scanf("%d", &n);

    int **dist = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++){
        dist[i] = (int*)malloc(sizeof(int) * n);
        for (int j = 0; j < n; j++){
            scanf("%d", &dist[i][j]);
        }
    }

    int *greedyBestSol = (int*)malloc(sizeof(int) * n);
    int *currSol = (int*)malloc(sizeof(int) * n);
    long long greedyBestCost = LLONG_MAX, currCost;
    double minAlpha = 0.01, maxAlpha = 0.1;
    int GREEDY_ITER = 1000;
    for (int iter = 0; iter < GREEDY_ITER; iter++) {
        double alpha = maxAlpha - ((maxAlpha - minAlpha) * iter / (GREEDY_ITER - 1));
        candidateSolution(n, dist, currSol, &currCost, alpha);
        if (currCost < greedyBestCost) {
            greedyBestCost = currCost;
            for (int i = 0; i < n; i++) {
                greedyBestSol[i] = currSol[i];
            }
        }
    }
    
    int *hybridBestTour = (int*)malloc(sizeof(int) * n);
    int hybridBestCost;
    simulatedAnnealingHybrid(n, dist, greedyBestSol, hybridBestTour, &hybridBestCost);

    printf("%d\n", n);
    for (int i = 0; i < n; i++){
        printf("%d ", hybridBestTour[i] + 1);
    }
    free(greedyBestSol);
    free(currSol);
    free(hybridBestTour);
    for (int i = 0; i < n; i++){
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
