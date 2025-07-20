#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define MAX_VARS 200

int parent[MAX_VARS];
double weight[MAX_VARS];
char* names[MAX_VARS];
int varCount = 0;

int getIndex(char* name) {
    for (int i = 0; i < varCount; ++i) {
        if (strcmp(names[i], name) == 0) return i;
    }
    names[varCount] = strdup(name);
    return varCount++;
}

int find(int x) {
    if (parent[x] != x) {
        int orig = parent[x];
        parent[x] = find(parent[x]);
        weight[x] *= weight[orig];
    }
    return parent[x];
}

bool checkContradictions(char*** equations, int equationsSize, int* equationsColSize, double* values, int valuesSize) {
    varCount = 0;

    for (int i = 0; i < MAX_VARS; ++i) {
        parent[i] = i;
        weight[i] = 1.0;
        names[i] = NULL;
    }

    for (int i = 0; i < equationsSize; ++i) {
        char* a = equations[i][0];
        char* b = equations[i][1];
        double val = values[i];

        int idxA = getIndex(a);
        int idxB = getIndex(b);

        int rootA = find(idxA);
        int rootB = find(idxB);

        if (rootA == rootB) {
            double expected = weight[idxA] / weight[idxB];
            if (fabs(expected - val) > 1e-5)
                return true;
        } else {
            parent[rootA] = rootB;
            weight[rootA] = val * weight[idxB] / weight[idxA];
        }
    }

    return false;
}
