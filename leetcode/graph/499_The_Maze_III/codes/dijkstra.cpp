#include <vector>
#include <string>
#include <queue>
#include <tuple>
#include <climits>
using namespace std;

class Solution {
public:
    string findShortestWay(vector<vector<int>>& maze, vector<int>& ball, vector<int>& hole) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
        vector<vector<string>> path(m, vector<string>(n, ""));

        priority_queue<Node, vector<Node>, greater<Node>> pq;
        pq.push({0, "", ball[0], ball[1]});
        dist[ball[0]][ball[1]] = 0;

        vector<vector<int>> dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        string dirStr = "durl";

        while (!pq.empty()) {
            auto [d, p, x, y] = pq.top(); pq.pop();
            if (x == hole[0] && y == hole[1]) return p;

            for (int i = 0; i < 4; ++i) {
                int nx = x, ny = y, steps = 0;
                while (isValid(nx + dirs[i][0], ny + dirs[i][1], maze)) {
                    nx += dirs[i][0];
                    ny += dirs[i][1];
                    steps++;
                    if (nx == hole[0] && ny == hole[1]) break;
                }
                string newPath = p + dirStr[i];
                if (dist[nx][ny] > d + steps || 
                    (dist[nx][ny] == d + steps && newPath < path[nx][ny])) {
                    dist[nx][ny] = d + steps;
                    path[nx][ny] = newPath;
                    pq.push({dist[nx][ny], newPath, nx, ny});
                }
            }
        }
        return "impossible";
    }

private:
    struct Node {
        int d;
        string p;
        int x, y;
        bool operator>(const Node& other) const {
            if (d != other.d) return d > other.d;
            return p > other.p;
        }
    };

    bool isValid(int x, int y, vector<vector<int>>& maze) {
        return x >= 0 && y >= 0 && x < maze.size() && y < maze[0].size() && maze[x][y] == 0;
    }
};
