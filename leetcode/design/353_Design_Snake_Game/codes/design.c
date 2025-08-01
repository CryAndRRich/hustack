#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int x, y;
    struct Node* next;
} Node;

typedef struct {
    int width, height;
    int** food;
    int foodSize;
    int foodIndex;
    int score;

    Node* head;
    Node* tail;
} SnakeGame;

SnakeGame* snakeGameCreate(int width, int height, int** food, int foodSize, int* foodColSize) {
    SnakeGame* obj = malloc(sizeof(SnakeGame));
    obj -> width = width;
    obj -> height = height;
    obj -> food = food;
    obj -> foodSize = foodSize;
    obj -> foodIndex = 0;
    obj -> score = 0;

    Node* node = malloc(sizeof(Node));
    node -> x = 0; node -> y = 0; node -> next = NULL;
    obj -> head = obj -> tail = node;

    return obj;
}

int collide(SnakeGame* obj, int x, int y) {
    for (Node* p = obj -> head; p; p = p -> next) {
        if (p -> x == x && p -> y == y && (p != obj -> tail))
            return 1;
    }
    return 0;
}

int snakeGameMove(SnakeGame* obj, char* direction) {
    int dx = 0, dy = 0;
    if (direction[0] == 'U') dx = -1;
    else if (direction[0] == 'D') dx = 1;
    else if (direction[0] == 'L') dy = -1;
    else if (direction[0] == 'R') dy = 1;

    int nx = obj -> head -> x + dx;
    int ny = obj -> head -> y + dy;

    if (nx < 0 || ny < 0 || nx >= obj -> height || ny >= obj -> width || collide(obj, nx, ny))
        return -1;

    Node* newHead = malloc(sizeof(Node));
    newHead -> x = nx; newHead -> y = ny;
    newHead -> next = obj -> head;
    obj -> head = newHead;

    if (obj -> foodIndex < obj -> foodSize && obj -> food[obj -> foodIndex][0] == nx && obj -> food[obj -> foodIndex][1] == ny) {
        obj -> score++;
        obj -> foodIndex++;
    } else {
        Node* p = obj -> head;
        while (p -> next && p -> next != obj -> tail) p = p -> next;
        free(obj -> tail);
        obj -> tail = p;
        obj -> tail -> next = NULL;
    }

    return obj -> score;
}

void snakeGameFree(SnakeGame* obj) {
    Node* p = obj -> head;
    while (p) {
        Node* tmp = p;
        p = p -> next;
        free(tmp);
    }
    free(obj);
}


/**
 * Your SnakeGame struct will be instantiated and called as such:
 * SnakeGame* obj = snakeGameCreate(width, height, food, foodSize, foodColSize);
 * int param_1 = snakeGameMove(obj, direction);
 
 * snakeGameFree(obj);
*/