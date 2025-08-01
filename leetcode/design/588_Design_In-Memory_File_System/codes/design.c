#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    char* name;
    int isFile;
    char* content;
    struct Node** children;
    int childrenSize;
    int childrenCapacity;
} Node;

typedef struct {
    Node* root;
} FileSystem;

Node* createNode(const char* name, int isFile) {
    Node* node = malloc(sizeof(Node));
    node -> name = strdup(name);
    node -> isFile = isFile;
    node -> content = isFile ? strdup("") : NULL;
    node -> children = NULL;
    node -> childrenSize = 0;
    node -> childrenCapacity = 0;
    return node;
}

Node* findOrCreate(Node* root, char* name, int isFile) {
    for (int i = 0; i < root -> childrenSize; ++i) {
        if (strcmp(root -> children[i] -> name, name) == 0) {
            return root -> children[i];
        }
    }

    if (root -> childrenSize == root -> childrenCapacity) {
        root -> childrenCapacity = root -> childrenCapacity == 0 ? 4 : root -> childrenCapacity * 2;
        root -> children = realloc(root -> children, sizeof(Node*) * root -> childrenCapacity);
    }

    Node* child = createNode(name, isFile);
    root -> children[root -> childrenSize++] = child;
    return child;
}

Node* traverse(FileSystem* obj, const char* path, int createMissing, int createFile) {
    char* temp = strdup(path);
    char* token = strtok(temp, "/");
    Node* curr = obj -> root;

    while (token) {
        Node* next = NULL;
        for (int i = 0; i < curr -> childrenSize; ++i) {
            if (strcmp(curr -> children[i] -> name, token) == 0) {
                next = curr -> children[i];
                break;
            }
        }
        if (!next) {
            if (createMissing || (createFile && strtok(NULL, "/") == NULL)) {
                next = findOrCreate(curr, token, createFile && strtok(NULL, "/") == NULL);
            } else {
                free(temp);
                return NULL;
            }
        }
        curr = next;
        token = strtok(NULL, "/");
    }
    free(temp);
    return curr;
}

FileSystem* fileSystemCreate() {
    FileSystem* fs = malloc(sizeof(FileSystem));
    fs -> root = createNode("", 0);
    return fs;
}

char** fileSystemLs(FileSystem* obj, char* path, int* retSize) {
    Node* node = traverse(obj, path, 0, 0);
    if (!node) {
        *retSize = 0;
        return NULL;
    }

    if (node -> isFile) {
        char** res = malloc(sizeof(char*));
        res[0] = strdup(node -> name);
        *retSize = 1;
        return res;
    }

    *retSize = node -> childrenSize;
    char** res = malloc(sizeof(char*) * (*retSize));
    for (int i = 0; i < node -> childrenSize; ++i) {
        res[i] = strdup(node -> children[i] -> name);
    }

    // Sort lexicographically
    for (int i = 0; i < *retSize; ++i) {
        for (int j = i + 1; j < *retSize; ++j) {
            if (strcmp(res[i], res[j]) > 0) {
                char* tmp = res[i];
                res[i] = res[j];
                res[j] = tmp;
            }
        }
    }

    return res;
}

void fileSystemMkdir(FileSystem* obj, char* path) {
    traverse(obj, path, 1, 0);
}

void fileSystemAddContentToFile(FileSystem* obj, char* filePath, char* content) {
    Node* fileNode = traverse(obj, filePath, 1, 1);
    if (!fileNode -> isFile) {
        fileNode -> isFile = 1;
        fileNode -> content = strdup("");
    }

    char* newContent = malloc(strlen(fileNode -> content) + strlen(content) + 1);
    strcpy(newContent, fileNode -> content);
    strcat(newContent, content);
    free(fileNode -> content);
    fileNode -> content = newContent;
}

char* fileSystemReadContentFromFile(FileSystem* obj, char* filePath) {
    Node* fileNode = traverse(obj, filePath, 0, 0);
    return fileNode ? fileNode -> content : NULL;
}

void freeNode(Node* node) {
    free(node -> name);
    if (node -> isFile) free(node -> content);
    for (int i = 0; i < node -> childrenSize; ++i) {
        freeNode(node -> children[i]);
    }
    free(node -> children);
    free(node);
}

void fileSystemFree(FileSystem* obj) {
    if (!obj) return;
    freeNode(obj -> root);
    free(obj);
}


/**
 * Your FileSystem struct will be instantiated and called as such:
 * FileSystem* obj = fileSystemCreate();
 * char** param_1 = fileSystemLs(obj, path, retSize);
 
 * fileSystemMkdir(obj, path);
 
 * fileSystemAddContentToFile(obj, filePath, content);
 
 * char* param_4 = fileSystemReadContentFromFile(obj, filePath);
 
 * fileSystemFree(obj);
*/