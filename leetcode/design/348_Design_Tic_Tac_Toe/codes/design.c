#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int* rows;
    int* cols;
    int diag;
    int anti_diag;
    int n;
} TicTacToe;

TicTacToe* ticTacToeCreate(int n) {
    TicTacToe* obj = (TicTacToe*)malloc(sizeof(TicTacToe));
    obj -> n = n;
    obj -> rows = (int*)calloc(n, sizeof(int));
    obj -> cols = (int*)calloc(n, sizeof(int));
    obj -> diag = 0;
    obj -> anti_diag = 0;
    return obj;
}

int ticTacToeMove(TicTacToe* obj, int row, int col, int player) {
    int mark = (player == 1) ? 1 : -1;
    obj -> rows[row] += mark;
    obj -> cols[col] += mark;
    if (row == col) obj -> diag += mark;
    if (row + col == obj -> n - 1) obj -> anti_diag += mark;

    if (abs(obj -> rows[row]) == obj -> n || abs(obj -> cols[col]) == obj -> n ||
        abs(obj -> diag) == obj -> n || abs(obj -> anti_diag) == obj -> n)
        return player;
    return 0;
}

void ticTacToeFree(TicTacToe* obj) {
    free(obj -> rows);
    free(obj -> cols);
    free(obj);
}


/**
 * Your TicTacToe struct will be instantiated and called as such:
 * TicTacToe* obj = ticTacToeCreate(n);
 * int param_1 = ticTacToeMove(obj, row, col, player);
 
 * ticTacToeFree(obj);
*/