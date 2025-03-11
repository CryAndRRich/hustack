#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define MAX 1000
#define INF 1000000000

#define GREEDY_ALPHA 0.05

#define NUM_ACO_ITERATIONS 500
#define ANT_COUNT 50
#define ALPHA 1.0  
#define BETA 5.0  
#define RHO 0.5   
#define Q 100.0  

#define SA_INITIAL_TEMP_FACTOR 10.0
#define SA_COOLING_RATE 0.99999
#define SA_T_MIN 1e-3
#define SA_MAX_ITER 10000000
#define SA_NO_IMPROVEMENT_THRESHOLD 100000

int n;
int **dist;
int computeCost(int n, int **dist, int *tour) {
    int cost = 0;
    for (int i = 0; i < n; i++) {
        cost += dist[tour[i]][tour[(i + 1) % n]];
    }
    return cost;
}

void candidateSolution(int n, int **dist, int *solution, long long *solCost, double alpha) {
    int curr = rand() % n;
    int count = 0;
    long long cost = 0;
    int *visited = (int*)calloc(n, sizeof(int));
    visited[curr] = 1;
    solution[count] = curr;
    count++;
    while (count < n) {
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
        if (numCandidates == 0) break;
        double minDist = candDist[0], maxDist = candDist[0];
        for (int j = 0; j < numCandidates; j++) {
            if (candDist[j] < minDist) minDist = candDist[j];
            if (candDist[j] > maxDist) maxDist = candDist[j];
        }
        double threshold = minDist + alpha * (maxDist - minDist);
        int candidateList[numCandidates];
        int candidateCount = 0;
        for (int j = 0; j < numCandidates; j++) {
            if (candDist[j] <= threshold) {
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

double **allocateDoubleMatrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void freeDoubleMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int chooseNextCity(int currCity, int n, int **dist, double **pheromone, double **heuristic, int *visited) {
    double sum = 0.0;
    double probabilities[MAX] = {0};
    for (int j = 0; j < n; j++) {
        if (!visited[j]) {
            double tau = pow(pheromone[currCity][j], ALPHA);
            double eta = pow(heuristic[currCity][j], BETA);
            probabilities[j] = tau * eta;
            sum += probabilities[j];
        }
    }
    double r = ((double)rand() / RAND_MAX) * sum;
    double accum = 0.0;
    for (int j = 0; j < n; j++) {
        if (!visited[j]) {
            accum += probabilities[j];
            if (accum >= r)
                return j;
        }
    }
    for (int j = 0; j < n; j++) {
        if (!visited[j])
            return j;
    }
    return -1;
}

void ACOPhase(int n, int **dist, int *bestTour, int *bestCost, int *seedTour, int seedCost) {
    double **pheromone = allocateDoubleMatrix(n);
    double **heuristic = allocateDoubleMatrix(n);
    double tau0 = 1.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            pheromone[i][j] = tau0;
            heuristic[i][j] = (dist[i][j] > 0) ? 1.0 / dist[i][j] : 0.0;
        }
    }
    double extra = Q / ((double)seedCost);
    for (int i = 0; i < n; i++){
        int from = seedTour[i];
        int to = seedTour[(i+1) % n];
        pheromone[from][to] += extra;
        pheromone[to][from] += extra;
    }
    int localBestTour[MAX];
    int localBestCost = INF;
    for (int iter = 0; iter < NUM_ACO_ITERATIONS; iter++){
        for (int ant = 0; ant < ANT_COUNT; ant++){
            int tour[MAX];
            int visited[MAX] = {0};
            int start = rand() % n;
            tour[0] = start;
            visited[start] = 1;
            for (int pos = 1; pos < n; pos++){
                int currCity = tour[pos - 1];
                int nextCity = chooseNextCity(currCity, n, dist, pheromone, heuristic, visited);
                tour[pos] = nextCity;
                visited[nextCity] = 1;
            }
            int cost = computeCost(n, dist, tour);
            if (cost < localBestCost) {
                localBestCost = cost;
                for (int i = 0; i < n; i++){
                    localBestTour[i] = tour[i];
                }
            }
            double delta = Q / (double)cost;
            for (int i = 0; i < n; i++){
                int from = tour[i];
                int to = tour[(i+1) % n];
                pheromone[from][to] += delta;
                pheromone[to][from] += delta;
            }
        }
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                pheromone[i][j] *= (1 - RHO);
            }
        }
        for (int i = 0; i < n; i++){
            int from = seedTour[i];
            int to = seedTour[(i+1) % n];
            pheromone[from][to] += extra;
            pheromone[to][from] += extra;
        }
    }
    *bestCost = localBestCost;
    for (int i = 0; i < n; i++){
        bestTour[i] = localBestTour[i];
    }
    freeDoubleMatrix(pheromone, n);
    freeDoubleMatrix(heuristic, n);
}

void SAPhase(int n, int **dist, int *initTour, int *bestTour, int *bestCost) {
    int currTour[MAX];
    for (int i = 0; i < n; i++){
        currTour[i] = initTour[i];
        bestTour[i] = initTour[i];
    }
    int currCost = computeCost(n, dist, currTour);
    *bestCost = currCost;
    
    int maxD = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (dist[i][j] > maxD)
                maxD = dist[i][j];
        }
    }
    double temperature = SA_INITIAL_TEMP_FACTOR * maxD;
    int iter = 0, noImprovementCount = 0;
    while (temperature > SA_T_MIN && iter < SA_MAX_ITER) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            int temp = currTour[i];
            currTour[i] = currTour[j];
            currTour[j] = temp;
            int newCost = computeCost(n, dist, currTour);
            int delta = newCost - currCost;
            if (delta < 0 || (exp(-delta / temperature) > ((double)rand() / RAND_MAX))) {
                currCost = newCost;
                if (currCost < *bestCost) {
                    *bestCost = currCost;
                    for (int k = 0; k < n; k++){
                        bestTour[k] = currTour[k];
                    }
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                }
            } else {
                temp = currTour[i];
                currTour[i] = currTour[j];
                currTour[j] = temp;
                noImprovementCount++;
            }
        }
        iter++;
        if (noImprovementCount > SA_NO_IMPROVEMENT_THRESHOLD) {
            temperature *= 1.1; 
            noImprovementCount = 0;
        } else {
            temperature *= SA_COOLING_RATE;
        }
    }
}

int main(){
    srand(time(NULL));
    scanf("%d", &n);
    
    dist = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++){
        dist[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++){
            scanf("%d", &dist[i][j]);
        }
    }
    
    int *greedyTour = (int*)malloc(n * sizeof(int));
    long long greedyCost;
    candidateSolution(n, dist, greedyTour, &greedyCost, GREEDY_ALPHA);

    int acoTour[MAX];
    int acoCost;
    ACOPhase(n, dist, acoTour, &acoCost, greedyTour, (int)greedyCost);

    int finalTour[MAX];
    int finalCost;
    SAPhase(n, dist, acoTour, finalTour, &finalCost);
    
    printf("%d\n", n);
    for (int i = 0; i < n; i++){
        printf("%d ", finalTour[i] + 1);
    }
    
    free(greedyTour);
    for (int i = 0; i < n; i++){
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
