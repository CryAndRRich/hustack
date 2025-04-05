#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define MAX 21

int n;                      
bool graph1[MAX][MAX];        
bool graph2[MAX][MAX]; 
int mp[MAX];    
bool visited[MAX];  

void compute_T_sets(bool T1[], bool T2[]) {
    int i, u, v;
    for(i = 0; i < n; i++){
        T1[i] = false;
        T2[i] = false;
    }
    for(u = 0; u < n; u++){
        if(mp[u] != -1) {
            for(v = 0; v < n; v++){
                if(graph1[u][v] && mp[v] == -1)
                    T1[v] = true;
            }
        }
    }
    for(u = 0; u < n; u++){
        if(mp[u] != -1) {
            int w = mp[u];
            for(v = 0; v < n; v++){
                if(graph2[w][v] && !visited[v])
                    T2[v] = true;
            }
        }
    }
}

bool feasible(int u, int v) {
    int i;
    for(i = 0; i < n; i++){
        if(mp[i] != -1) {
            if(graph1[u][i] != graph2[v][mp[i]])
                return false;
        }
    }
    return true;
}

bool match(int depth) {
    if(depth == n)
        return true;

    bool T1[MAX], T2[MAX];
    compute_T_sets(T1, T2);

    int candidate_u = -1;
    bool found_T1 = false;
    int u;
    for(u = 0; u < n; u++){
        if(mp[u] == -1 && T1[u]){
            candidate_u = u;
            found_T1 = true;
            break;
        }
    }
    if(candidate_u == -1) { 
        for(u = 0; u < n; u++){
            if(mp[u] == -1){
                candidate_u = u;
                break;
            }
        }
    }

    bool any_T2 = false;
    for(u = 0; u < n; u++){
        if(!visited[u] && T2[u]){
            any_T2 = true;
            break;
        }
    }
    
    int v;
    for(v = 0; v < n; v++){
        if(!visited[v]) {
            if(any_T2 && !T2[v])
                continue;
            if(feasible(candidate_u, v)){
                mp[candidate_u] = v;
                visited[v] = true;
                if(match(depth + 1))
                    return true;
                mp[candidate_u] = -1;
                visited[v] = false;
            }
        }
    }
    return false;
}

int main(){
    int n1, n2, m1, m2, i, u, v;
    scanf("%d %d", &n1, &m1);
    
    for (int i = 0; i < n; i += 1) {
        for (int j = 0; j < n; j += 1) {
            graph1[i][j] = false;
            graph2[i][j] = false;
        }
    }
    
    for (int i = 0; i < m1; i += 1) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph1[u - 1][v - 1] = true;
        graph1[v - 1][u - 1] = true;
    }
    
    scanf("%d %d", &n2, &m2);
    if (n1 != n2) {
        printf("%d", 0);
        return 0;
    }
    n = n1;
    for (int i = 0; i < m2; i += 1) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph2[u - 1][v - 1] = true;
        graph2[v - 1][u - 1] = true;
    }
    
    for(i = 0; i < n; i++){
        mp[i] = -1;
        visited[i] = false;
    }
    
    bool iso = match(0);
    printf("%d", iso ? 1 : 0);
    return 0;
}
