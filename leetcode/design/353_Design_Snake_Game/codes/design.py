from collections import deque

class SnakeGame:

    def __init__(self, width: int, height: int, food: list[list[int]]):
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.score = 0

        self.snake = deque([(0, 0)])
        self.snake_set = {(0, 0)}

        self.dirs = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    def move(self, direction: str) -> int:
        dx, dy = self.dirs[direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if (not (0 <= new_head[0] < self.height) or
            not (0 <= new_head[1] < self.width) or
            (new_head in self.snake_set and new_head != self.snake[-1])):
            return -1

        if self.food_index < len(self.food) and self.food[self.food_index] == list(new_head):
            self.food_index += 1
            self.score += 1
        else:
            tail = self.snake.pop()
            self.snake_set.remove(tail)

        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)

        return self.score

# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)