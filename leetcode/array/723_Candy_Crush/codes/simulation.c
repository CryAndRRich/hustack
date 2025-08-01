/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

int** candyCrush(int** board, int boardSize, int* boardColSize, int* returnSize, int** returnColumnSizes) {
    int m = boardSize, n = boardColSize[0];
    bool changed = true;

    while (changed) {
        changed = false;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n - 2; j++) {
                int v = abs(board[i][j]);
                if (v != 0 && abs(board[i][j + 1]) == v && abs(board[i][j + 2]) == v) {
                    board[i][j] = board[i][j + 1] = board[i][j + 2] = -v;
                    changed = true;
                }
            }

        for (int i = 0; i < m - 2; i++)
            for (int j = 0; j < n; j++) {
                int v = abs(board[i][j]);
                if (v != 0 && abs(board[i + 1][j]) == v && abs(board[i + 2][j]) == v) {
                    board[i][j] = board[i + 1][j] = board[i + 2][j] = -v;
                    changed = true;
                }
            }

        for (int j = 0; j < n; j++) {
            int write = m - 1;
            for (int i = m - 1; i >= 0; i--)
                if (board[i][j] > 0)
                    board[write--][j] = board[i][j];
            for (int i = write; i >= 0; i--)
                board[i][j] = 0;
        }
    }

    *returnSize = boardSize;
    *returnColumnSizes = (int*)malloc(sizeof(int) * boardSize);
    for (int i = 0; i < boardSize; i++)
        (*returnColumnSizes)[i] = boardColSize[0];
    return board;
}
