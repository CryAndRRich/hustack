#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int to;
    struct Node* next;
} Node;

Node** createGraph(int n) {
    Node** graph = (Node**)calloc(n, sizeof(Node*));
    return graph;
}

void addEdge(Node** graph, int u, int v) {
    Node* node = (Node*)malloc(sizeof(Node));
    node -> to = v;
    node -> next = graph[u];
    graph[u] = node;
}

void freeGraph(Node** graph, int n) {
    for (int i = 0; i < n; i++) {
        Node* cur = graph[i];
        while (cur) {
            Node* tmp = cur;
            cur = cur -> next;
            free(tmp);
        }
    }
    free(graph);
}

typedef struct Stack {
    int* data;
    int size;
    int capacity;
} Stack;

Stack* createStack(int capacity) {
    Stack* s = (Stack*)malloc(sizeof(Stack));
    s -> data = (int*)malloc(sizeof(int)*capacity);
    s -> size = 0;
    s -> capacity = capacity;
    return s;
}

void push(Stack* s, int val) {
    if (s -> size == s -> capacity) {
        s -> capacity *= 2;
        s -> data = (int*)realloc(s -> data, sizeof(int)*s -> capacity);
    }
    s -> data[s -> size++] = val;
}

int pop(Stack* s) {
    return s -> data[--s -> size];
}

int isEmpty(Stack* s) {
    return s -> size == 0;
}

void tarjan(int u, Node** graph, int* index, int* lowlink, int* onStack, Stack* stack, int* idx, int* scc_id, int* scc_count) {
    index[u] = (*idx);
    lowlink[u] = (*idx);
    (*idx)++;
    push(stack, u);
    onStack[u] = 1;

    for (Node* p = graph[u]; p != NULL; p = p -> next) {
        int v = p -> to;
        if (index[v] == 0) {
            tarjan(v, graph, index, lowlink, onStack, stack, idx, scc_id, scc_count);
            if (lowlink[v] < lowlink[u]) lowlink[u] = lowlink[v];
        }
        else if (onStack[v]) {
            if (index[v] < lowlink[u]) lowlink[u] = index[v];
        }
    }

    if (lowlink[u] == index[u]) {
        int node;
        do {
            node = pop(stack);
            onStack[node] = 0;
            scc_id[node] = *scc_count;
        } while (node != u);
        (*scc_count)++;
    }
}

int minRunesToAdd(int n, int* crystals, int crystalsSize, int* flowFrom, int flowFromSize, int* flowTo, int flowToSize) {
    Node** graph = createGraph(n);
    for (int i = 0; i < flowFromSize; i++) {
        addEdge(graph, flowFrom[i], flowTo[i]);
    }

    int* index = (int*)calloc(n, sizeof(int));
    int* lowlink = (int*)calloc(n, sizeof(int));
    int* onStack = (int*)calloc(n, sizeof(int));
    int* scc_id = (int*)malloc(sizeof(int)*n);
    memset(scc_id, -1, sizeof(int)*n);
    Stack* stack = createStack(n);

    int idx = 1, scc_count = 0;
    for (int i = 0; i < n; i++) {
        if (index[i] == 0) {
            tarjan(i, graph, index, lowlink, onStack, stack, &idx, scc_id, &scc_count);
        }
    }

    int* indegree = (int*)calloc(scc_count, sizeof(int));
    for (int u = 0; u < n; u++) {
        for (Node* p = graph[u]; p != NULL; p = p -> next) {
            int v = p -> to;
            if (scc_id[u] != scc_id[v]) {
                indegree[scc_id[v]]++;
            }
        }
    }

    char* crystalScc = (char*)calloc(scc_count, sizeof(char));
    for (int i = 0; i < crystalsSize; i++) {
        crystalScc[scc_id[crystals[i]]] = 1;
    }

    int result = 0;
    for (int i = 0; i < scc_count; i++) {
        if (indegree[i] == 0 && !crystalScc[i]) {
            result++;
        }
    }

    free(index);
    free(lowlink);
    free(onStack);
    free(scc_id);
    free(indegree);
    free(crystalScc);
    freeGraph(graph, n);
    free(stack -> data);
    free(stack);

    return result;
}
