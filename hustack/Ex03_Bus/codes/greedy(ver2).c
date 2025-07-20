#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

void candidateSolution(int n, int k, int **dist, int *solution, long long *solCost, double alpha) {
    int curr = 0;  
    int load = 0;
    int count = 0;

    int *picked = (int*)calloc(n+1, sizeof(int));
    int *dropped = (int*)calloc(n+1, sizeof(int));
    long long cost = 0;
    
    while(count < 2 * n) {
        int numCandidates = 0;
        int candidates[2*n];
        double candDist[2*n];
         
        
        for (int i = 1; i <= n; i++) {
            if(!picked[i] && load < k) {
                candidates[numCandidates] = i;
                candDist[numCandidates] = dist[curr][i];
                numCandidates++;
            }
        }
        
        for (int i = 1; i <= n; i++) {
            if(picked[i] && !dropped[i]) {
                candidates[numCandidates] = i + n;
                candDist[numCandidates] = dist[curr][i+n];
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
         
        int candidateList[2*n];
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
        if(chosen <= n) { 
            picked[chosen] = 1;
            load++;
        } else { 
            int passenger = chosen - n;
            dropped[passenger] = 1;
            load--;
        }
        count++;
    }
    cost += dist[curr][0]; 
    *solCost = cost;
    free(picked);
    free(dropped);
}

int main(){
    int n, k;
    scanf("%d %d", &n, &k);
    int size = 2 * n + 1;

    int **dist = (int**)malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++){
        dist[i] = (int*)malloc(sizeof(int) * size);
        for (int j = 0; j < size; j++){
            scanf("%d", &dist[i][j]);
        }
    }
    
    srand(time(NULL));
    
    int *bestSol = (int*)malloc(sizeof(int) * 2 * n);
    long long bestCost = 1000000000000000000LL;
    int *currSol = (int*)malloc(sizeof(int) * 2 * n);
    long long currCost;
    
    double minAlpha = 0.01, maxAlpha = 0.1;
    
    for (int iter = 0; iter < 1000; iter++) {
        double alpha = maxAlpha - ((maxAlpha - minAlpha) * iter / (double)(999));
        candidateSolution(n, k, dist, currSol, &currCost, alpha);
        if(currCost < bestCost) {
            bestCost = currCost;
            for (int i = 0; i < 2 * n; i++) {
                bestSol[i] = currSol[i];
            }
        }
    }
    
    printf("%d\n", n);
    for (int i = 0; i < 2 * n; i++){
        printf("%d ", bestSol[i]);
    }
    
    free(bestSol);
    free(currSol);
    for (int i = 0; i < size; i++){
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
