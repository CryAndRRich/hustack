#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

int n, k;
int **dist;
int best_cost = INT_MAX;
int *best_path;
int best_path_len = 0;

int predict(bool visited[], int curr) {
    int min_cost = 0;
    for (int i = 1; i <= 2 * n; i++) {
        if (!visited[i]) {
            int temp = dist[curr][i];
            if (temp < 0) temp = 0; 
            min_cost += temp;
        }
    }
    return min_cost;
}

void copy_path(int *end, int *start, int len, int last) {
    for (int i = 0; i < len; i++) {
        end[i] = start[i];
    }
    end[len] = last;
}

void backtrack(int curr, int load, int visited_count, bool visited[], int curr_cost, int path[], int path_len) {
    if (visited_count == 2 * n) {
        int total_cost = curr_cost + dist[curr][0];  
        if (total_cost < best_cost) {
            best_cost = total_cost;
            best_path_len = path_len;
            copy_path(best_path, path, path_len, 0);
        }
        return;
    }
    
    int pred = curr_cost + predict(visited, curr);
    if (pred >= best_cost)
        return; 
        
    for (int i = 1; i <= 2 * n; i++) {
        if (!visited[i]) {
            if (i <= n) {
                if (load < k) {
                    visited[i] = true;
                    path[path_len] = i;
                    backtrack(i, load + 1, visited_count + 1, visited, curr_cost + dist[curr][i], path, path_len + 1);
                    visited[i] = false;
                }
            }
            else {
                if (visited[i - n]) {
                    visited[i] = true;
                    path[path_len] = i;
                    backtrack(i, load - 1, visited_count + 1, visited, curr_cost + dist[curr][i], path, path_len + 1);
                    visited[i] = false;
                }
            }
        }
    }
}

int main(){
    scanf("%d %d", &n, &k);
    int size = 2 * n + 1;
    dist = (int**)malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++) {
        dist[i] = (int*)malloc(sizeof(int) * size);
        for (int j = 0; j < size; j++) {
            scanf("%d", &dist[i][j]);
        }
    }
    
    bool *visited = (bool*)malloc(sizeof(bool) * (2 * n + 1));
    for (int i = 0; i < 2 * n + 1; i++) visited[i] = false;
    
    int *path = (int*)malloc(sizeof(int) * (2 * n + 2));
    best_path = (int*)malloc(sizeof(int) * (2 * n + 2));
    
    path[0] = 0;
    backtrack(0, 0, 0, visited, 0, path, 1);
    
    printf("%d\n", n);
    for (int i = 1; i < best_path_len; i++) { 
        printf("%d ", best_path[i]);
    }
    
    for (int i = 0; i < size; i++)
        free(dist[i]);
    free(dist);
    free(visited);
    free(path);
    free(best_path);
    
    return 0;
}
