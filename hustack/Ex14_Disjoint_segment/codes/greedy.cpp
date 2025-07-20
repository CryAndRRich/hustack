#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

bool cmp(const vector<int>& a, const vector<int>& b) {
    if(a[1] == b[1])
        return a[0] < b[0];
    return a[1] < b[1];
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<vector<int>> segments(n, vector<int>(2));
    
    for (int i = 0; i < n; i++){
        cin >> segments[i][0] >> segments[i][1];
    }
    
    sort(segments.begin(), segments.end(), cmp);
    
    int count = 0;
    int last_end = -1;
    for (int i = 0; i < n; i++){
        if(segments[i][0] > last_end){
            count += 1;
            last_end = segments[i][1];
        }
    }
    
    cout << count;
    return 0;
}
