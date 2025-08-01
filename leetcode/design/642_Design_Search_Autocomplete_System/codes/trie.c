#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define CHARSET 128   // extended ASCII
#define MAX_SENTENCE_LEN 100
#define MAX_RETURN 3

typedef struct TrieNode {
    struct TrieNode* children[CHARSET];
    int times;
    bool isEnd;
    char* sentence;
} TrieNode;

typedef struct {
    TrieNode* root;
    TrieNode* cur;
    char buffer[201];
    int bufLen;
} AutocompleteSystem;

typedef struct {
    char* sentence;
    int times;
} SentenceCount;

// ===== Priority Queue Helpers =====
int compare(const void* a, const void* b) {
    SentenceCount* sa = (SentenceCount*)a;
    SentenceCount* sb = (SentenceCount*)b;
    if (sa -> times != sb -> times) return sb -> times - sa -> times;
    return strcmp(sa -> sentence, sb -> sentence);
}

// ===== Trie Helpers =====
TrieNode* newTrieNode() {
    TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
    return node;
}

void insert(TrieNode* root, const char* sentence, int times) {
    TrieNode* node = root;
    for (int i = 0; sentence[i]; i++) {
        char c = sentence[i];
        if (!node -> children[(int)c]) {
            node -> children[(int)c] = newTrieNode();
        }
        node = node -> children[(int)c];
    }
    node -> isEnd = true;
    node -> times += times;
    if (!node -> sentence) node -> sentence = strdup(sentence);
}

void dfs(TrieNode* node, SentenceCount* arr, int* idx) {
    if (!node) return;
    if (node -> isEnd) {
        arr[*idx].sentence = node -> sentence;
        arr[*idx].times = node -> times;
        (*idx)++;
    }
    for (int i = 0; i < CHARSET; i++) {
        if (node -> children[i]) {
            dfs(node -> children[i], arr, idx);
        }
    }
}

// ===== Autocomplete System =====
AutocompleteSystem* autocompleteSystemCreate(char** sentences, int sentencesSize, int* times, int timesSize) {
    AutocompleteSystem* sys = (AutocompleteSystem*)malloc(sizeof(AutocompleteSystem));
    sys -> root = newTrieNode();
    sys -> cur = sys -> root;
    sys -> bufLen = 0;

    for (int i = 0; i < sentencesSize; i++) {
        insert(sys -> root, sentences[i], times[i]);
    }
    return sys;
}

char** autocompleteSystemInput(AutocompleteSystem* obj, char c, int* returnSize) {
    if (c == '#') {
        obj -> buffer[obj -> bufLen] = '\0';
        insert(obj -> root, obj -> buffer, 1);
        obj -> cur = obj -> root;
        obj -> bufLen = 0;
        *returnSize = 0;
        return NULL;
    }

    obj -> buffer[obj -> bufLen++] = c;

    if (obj -> cur && obj -> cur -> children[(int)c]) {
        obj -> cur = obj -> cur -> children[(int)c];
    } else {
        obj -> cur = NULL;
        *returnSize = 0;
        return NULL;
    }

    SentenceCount temp[1000];
    int count = 0;
    dfs(obj -> cur, temp, &count);
    qsort(temp, count, sizeof(SentenceCount), compare);

    int top = count < MAX_RETURN ? count : MAX_RETURN;
    char** result = (char**)malloc(sizeof(char*) * top);
    for (int i = 0; i < top; i++) {
        result[i] = strdup(temp[i].sentence);
    }
    *returnSize = top;
    return result;
}

void trieFree(TrieNode* node) {
    if (!node) return;
    for (int i = 0; i < CHARSET; i++) {
        trieFree(node -> children[i]);
    }
    if (node -> sentence) free(node -> sentence);
    free(node);
}

void autocompleteSystemFree(AutocompleteSystem* obj) {
    trieFree(obj -> root);
    free(obj);
}

/**
 * Your AutocompleteSystem struct will be instantiated and called as such:
 * AutocompleteSystem* obj = autocompleteSystemCreate(sentences, sentencesSize, times, timesSize);
 * char** param_1 = autocompleteSystemInput(obj, c, retSize);
 
 * autocompleteSystemFree(obj);
*/