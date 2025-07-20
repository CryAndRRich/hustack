#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS 500
#define MAX_LEN 6
#define ALPHABET_SIZE 26

typedef struct TrieNode {
    struct TrieNode* children[ALPHABET_SIZE];
    int wordIndices[MAX_WORDS];
    int wordCount;
} TrieNode;

char** g_words;
int g_wordLength;
int** g_resultsColumnSizes;
char*** g_results;
int g_resultsSize;
int g_resultsCapacity;

TrieNode* createTrieNode() {
    TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
    memset(node -> children, 0, sizeof(node -> children));
    node -> wordCount = 0;
    return node;
}

void insertTrie(TrieNode* root, char* word, int index) {
    TrieNode* node = root;
    for (int i = 0; word[i]; i++) {
        int c = word[i] - 'a';
        if (!node -> children[c]) {
            node -> children[c] = createTrieNode();
        }
        node = node -> children[c];
        node -> wordIndices[node -> wordCount++] = index;
    }
}

void getWordsWithPrefix(TrieNode* root, char* prefix, int* indices, int* returnSize) {
    TrieNode* node = root;
    for (int i = 0; prefix[i]; i++) {
        int c = prefix[i] - 'a';
        if (!node -> children[c]) {
            *returnSize = 0;
            return;
        }
        node = node -> children[c];
    }
    memcpy(indices, node -> wordIndices, node -> wordCount * sizeof(int));
    *returnSize = node -> wordCount;
}

void ensureResultCapacity() {
    if (g_resultsSize == g_resultsCapacity) {
        g_resultsCapacity *= 2;
        g_results = realloc(g_results, g_resultsCapacity * sizeof(char**));
        *g_resultsColumnSizes = realloc(*g_resultsColumnSizes, g_resultsCapacity * sizeof(int));
    }
}

void copySquare(char** square) {
    ensureResultCapacity();
    char** newSquare = (char**)malloc(g_wordLength * sizeof(char*));
    for (int i = 0; i < g_wordLength; ++i) {
        newSquare[i] = strdup(square[i]);
    }
    g_results[g_resultsSize] = newSquare;
    (*g_resultsColumnSizes)[g_resultsSize] = g_wordLength;
    g_resultsSize++;
}

void backtrack(int step, char** square, TrieNode* root) {
    if (step == g_wordLength) {
        copySquare(square);
        return;
    }

    char prefix[MAX_LEN] = {0};
    for (int i = 0; i < step; ++i) {
        prefix[i] = square[i][step];
    }
    prefix[step] = '\0';

    int indices[MAX_WORDS];
    int count;
    getWordsWithPrefix(root, prefix, indices, &count);

    for (int i = 0; i < count; ++i) {
        square[step] = g_words[indices[i]];
        backtrack(step + 1, square, root);
    }
}

char*** wordSquares(char** words, int wordsSize, int* returnSize, int** returnColumnSizes) {
    if (wordsSize == 0) {
        *returnSize = 0;
        *returnColumnSizes = NULL;
        return NULL;
    }

    g_words = words;
    g_wordLength = strlen(words[0]);

    g_resultsCapacity = 1000;
    g_resultsSize = 0;
    g_results = (char***)malloc(g_resultsCapacity * sizeof(char**));
    *returnColumnSizes = (int*)malloc(g_resultsCapacity * sizeof(int));
    g_resultsColumnSizes = returnColumnSizes;

    TrieNode* root = createTrieNode();
    for (int i = 0; i < wordsSize; ++i) {
        insertTrie(root, words[i], i);
    }

    char** square = (char**)malloc(g_wordLength * sizeof(char*));
    for (int i = 0; i < wordsSize; ++i) {
        square[0] = words[i];
        backtrack(1, square, root);
    }
    free(square);

    *returnSize = g_resultsSize;
    return g_results;
}
