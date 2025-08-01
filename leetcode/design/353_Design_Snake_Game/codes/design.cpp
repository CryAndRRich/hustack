#include <vector>
#include <deque>
#include <unordered_set>
#include <string>
using namespace std;

class SnakeGame {
    int width, height, score = 0, foodIndex = 0;
    vector<vector<int>> food;
    deque<pair<int, int>> snake;
    unordered_set<int> snakeSet;

    int hash(int x, int y) { return x * width + y; }

public:
    SnakeGame(int width, int height, vector<vector<int>>& food)
        : width(width), height(height), food(food) {
        snake.push_front({0, 0});
        snakeSet.insert(hash(0, 0));
    }

    int move(string direction) {
        int dx = 0, dy = 0;
        if (direction == "U") dx = -1;
        else if (direction == "D") dx = 1;
        else if (direction == "L") dy = -1;
        else if (direction == "R") dy = 1;

        int nx = snake.front().first + dx;
        int ny = snake.front().second + dy;
        int newHead = hash(nx, ny);

        if (nx < 0 || ny < 0 || nx >= height || ny >= width ||
            (snakeSet.count(newHead) && make_pair(nx, ny) != snake.back()))
            return -1;

        if (foodIndex < food.size() && food[foodIndex][0] == nx && food[foodIndex][1] == ny) {
            foodIndex++;
            score++;
        } else {
            auto tail = snake.back();
            snakeSet.erase(hash(tail.first, tail.second));
            snake.pop_back();
        }

        snake.push_front({nx, ny});
        snakeSet.insert(newHead);
        return score;
    }
};


/**
 * Your SnakeGame object will be instantiated and called as such:
 * SnakeGame* obj = new SnakeGame(width, height, food);
 * int param_1 = obj -> move(direction);
 */