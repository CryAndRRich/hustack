#include <stdlib.h>
#include <string.h>

#define MAX_PLAYERS 10001

typedef struct {
    int scores[MAX_PLAYERS];
} Leaderboard;

Leaderboard* leaderboardCreate() {
    Leaderboard* obj = (Leaderboard*)malloc(sizeof(Leaderboard));
    memset(obj -> scores, 0, sizeof(obj -> scores));
    return obj;
}

void leaderboardAddScore(Leaderboard* obj, int playerId, int score) {
    obj -> scores[playerId] += score;
}

int cmp(const void* a, const void* b) {
    return (*(int*)b) - (*(int*)a);
}

int leaderboardTop(Leaderboard* obj, int K) {
    int* temp = (int*)malloc(sizeof(int) * MAX_PLAYERS);
    memcpy(temp, obj -> scores, sizeof(obj -> scores));
    qsort(temp, MAX_PLAYERS, sizeof(int), cmp);
    int sum = 0;
    for (int i = 0; i < K; ++i)
        sum += temp[i];
    free(temp);
    return sum;
}

void leaderboardReset(Leaderboard* obj, int playerId) {
    obj -> scores[playerId] = 0;
}

void leaderboardFree(Leaderboard* obj) {
    free(obj);
}

/**
 * Your Leaderboard struct will be instantiated and called as such:
 * Leaderboard* obj = leaderboardCreate();
 * leaderboardAddScore(obj, playerId, score);
 
 * int param_2 = leaderboardTop(obj, K);
 
 * leaderboardReset(obj, playerId);
 
 * leaderboardFree(obj);
*/