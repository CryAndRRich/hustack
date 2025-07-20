#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace std;

void candidateSolution(int n, int k, const vector<vector<int>> &dist, vector<int> &solution, long long &solCost, double alpha) {
    int curr = 0;
    int load = 0;
    int count = 0;

    vector<int> picked(n + 1, 0);
    vector<int> dropped(n + 1, 0);
    long long cost = 0;
    
    while(count < 2 * n) {
        int numCandidates = 0;
        vector<int> candidates(2 * n);
        vector<double> candDist(2 * n);
        
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
        
        if(numCandidates == 0)
            break;
        
        double minDist = candDist[0], maxDist = candDist[0];
        for (int j = 0; j < numCandidates; j++) {
            if(candDist[j] < minDist) minDist = candDist[j];
            if(candDist[j] > maxDist) maxDist = candDist[j];
        }
        double threshold = minDist + alpha * (maxDist - minDist);
        
        vector<int> candidateList;
        for (int j = 0; j < numCandidates; j++) {
            if(candDist[j] <= threshold) {
                candidateList.push_back(j);
            }
        }
        
        int chosenIndex = candidateList[rand() % candidateList.size()];
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
    solCost = cost;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, k;
    cin >> n >> k;
    int size = 2 * n + 1;
    
    vector<vector<int>> dist(size, vector<int>(size));
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            cin >> dist[i][j];
        }
    }
    
    srand(time(NULL));
    
    vector<int> bestSol(2 * n, 0);
    long long bestCost = 1000000000000000000LL;
    vector<int> currSol(2 * n, 0);
    long long currCost;
    
    double minAlpha = 0.01, maxAlpha = 0.1;
    
    for (int iter = 0; iter < 1000; iter++) {
        double alpha = maxAlpha - ((maxAlpha - minAlpha) * iter / 999.0);
        candidateSolution(n, k, dist, currSol, currCost, alpha);
        if(currCost < bestCost) {
            bestCost = currCost;
            bestSol = currSol;
        }
    }
    
    cout << n << "\n";
    for (int i = 0; i < 2 * n; i++){
        cout << bestSol[i] << " ";
    }
    
    return 0;
}
