#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX 1000
#define INF 1000000000

#define NUM_ITERATIONS 1000
#define ANT_COUNT 50        
#define ALPHA 1.0 
#define BETA 5.0  
#define RHO 0.5  
#define Q 100.0  

void candidateSolution(int n, int **dist, int *solution, int *solCost, double alpha) {
    int curr = rand() % n;  
    int count = 0;
    int cost = 0;

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

double **allocateDoubleMatrix(int n) {
    double **matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++){
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void freeDoubleMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++){
        free(matrix[i]);
    }
    free(matrix);
}

int tourCost(int n, int **dist, int *tour) {
    int cost = 0;
    for (int i = 0; i < n; i++){
        cost += dist[tour[i]][tour[(i + 1) % n]];
    }
    return cost;
}

int chooseNextCity(int currCity, int n, int **dist, double **pheromone, double **heuristic, int *visited) {
    double sum = 0.0;
    double probabilities[MAX];
    for (int j = 0; j < n; j++){
        if (!visited[j]) {
            double tau = pow(pheromone[currCity][j], ALPHA);
            double eta = pow(heuristic[currCity][j], BETA);
            probabilities[j] = tau * eta;
            sum += probabilities[j];
        } else {
            probabilities[j] = 0.0;
        }
    }
    double r = ((double)rand() / RAND_MAX) * sum;
    double accum = 0.0;
    for (int j = 0; j < n; j++){
        if (!visited[j]){
            accum += probabilities[j];
            if (accum >= r) {
                return j;
            }
        }
    }
    for (int j = 0; j < n; j++){
        if (!visited[j])
            return j;
    }
    return -1;
}
int main(){
    srand(time(NULL));
    int n;
    scanf("%d", &n);
    
    int **dist = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++){
        dist[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++){
            scanf("%d", &dist[i][j]);
        }
    }
    
    int *greedyTour = (int*)malloc(n * sizeof(int));
    int greedyCost;
    double greedyAlpha = 0.05;
    candidateSolution(n, dist, greedyTour, &greedyCost, greedyAlpha);
    
    double **pheromone = allocateDoubleMatrix(n);
    double **heuristic = allocateDoubleMatrix(n);
    double tau0 = 1.0; 
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            pheromone[i][j] = tau0;
            if (dist[i][j] > 0)
                heuristic[i][j] = 1.0 / dist[i][j];
            else
                heuristic[i][j] = 0.0;
        }
    }
    double extra = Q / (double)greedyCost;
    for (int i = 0; i < n; i++){
        int from = greedyTour[i];
        int to = greedyTour[(i + 1) % n];
        pheromone[from][to] += extra;
        pheromone[to][from] += extra;
    }
    
    int bestTour[MAX];
    int bestCost = INF;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++){
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
            int cost = tourCost(n, dist, tour);
            if (cost < bestCost) {
                bestCost = cost;
                for (int i = 0; i < n; i++){
                    bestTour[i] = tour[i];
                }
            }
            double delta = Q / (double)cost;
            for (int i = 0; i < n; i++){
                int from = tour[i];
                int to = tour[(i + 1) % n];
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
            int from = greedyTour[i];
            int to = greedyTour[(i + 1) % n];
            pheromone[from][to] += extra;
            pheromone[to][from] += extra;
        }
    }
    
    printf("%d\n", n);
    for (int i = 0; i < n; i++){
        printf("%d ", bestTour[i] + 1);
    }
    
    free(greedyTour);
    freeDoubleMatrix(pheromone, n);
    freeDoubleMatrix(heuristic, n);
    for (int i = 0; i < n; i++){
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
