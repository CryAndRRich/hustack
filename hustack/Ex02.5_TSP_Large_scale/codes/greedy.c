#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

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

int main(){
    int n;
    scanf("%d", &n);

    int **dist = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++){
        dist[i] = (int*)malloc(sizeof(int) * n);
        for (int j = 0; j < n; j++){
            scanf("%d", &dist[i][j]);
        }
    }

    srand(time(NULL));
    
    int *bestSol = (int*)malloc(sizeof(int) * n);
    long long bestCost = LLONG_MAX;
    int *currSol = (int*)malloc(sizeof(int) * n);
    long long currCost;

    double minAlpha = 0.01, maxAlpha = 0.1;
    for (int iter = 0; iter < 1000; iter++) {
        double alpha = maxAlpha - ((maxAlpha - minAlpha) * iter / 999.0);
        candidateSolution(n, dist, currSol, &currCost, alpha);
        if (currCost < bestCost) {
            bestCost = currCost;
            for (int i = 0; i < n; i++) {
                bestSol[i] = currSol[i];
            }
        }
    }
    
    printf("%d\n", n);
    for (int i = 0; i < n; i++){
        printf("%d ", bestSol[i] + 1);
    }

    free(bestSol);
    free(currSol);
    for (int i = 0; i < n; i++){
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
