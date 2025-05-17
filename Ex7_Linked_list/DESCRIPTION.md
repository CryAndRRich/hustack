## Description
Write a program that performs the following tasks:

**Initialization:**  
Build a linked list with the keys provided initially as the sequence: $a_1, a_2, ..., a_n$

**Operations:**  
Then, perform the following operations on the list:
- **addlast k:** Add an element with key equal to $k$ to the end of the list (only if $k$ does not already exist).
- **addfirst k:** Add an element with key equal to $k$ to the beginning of the list (only if $k$ does not already exist).
- **addafter u v:** Add an element with key equal to $u$ after the element with key equal to $v$ in the list (only if $v$ exists in the list and $u$ does not exist).
- **addbefore u v:** Add an element with key equal to $u$ before the element with key equal to $v$ in the list (only if $v$ exists in the list and $u$ does not exist).
- **remove k:** Remove the element with key equal to $k$ from the list.
- **reverse:** Reverse the order of the elements in the list (without allocating new nodes, only by changing pointers).

## Input
- **Line 1:** A positive integer $n$ ($1 \leq n \leq 1000$)  
- **Line 2:** Contains the positive integers $a_1, a_2, ..., a_n$ separated by spaces.
- **Subsequent lines:** Each line is a command to manipulate the list (the sequence of commands terminates with the symbol #).

## Output
Print the keys of the final list (in order) after executing all the given commands.

## Time Complexity

**createNode:** $O(1)$ — Allocates memory and initializes a node.

**findNode:** $O(n)$ — In the worst case, it traverses the entire list of n nodes.

**addFirst:** $O(n)$ — It calls **findNode** to check for the existence of the key ($O(n)$ in the worst case); inserting at the head takes $O(1)$.

**addLast:** $O(n)$ — It calls **findNode** and then traverses the list to find the last node.

**addAfter:** $O(n)$ — It involves two **findNode** calls: one to check if the new key exists and one to find the node with the given afterKey (each $O(n)$ in worst-case).

**addBefore:** $O(n)$ — Similar to **addAfter**, it uses **findNode** and a traversal to locate the node before which insertion is done.

**removeNode:** $O(n)$ — It may traverse the entire list to locate the node to remove.

**reverseList:** $O(n)$ — It traverses the list once to reverse the pointers.

**Overall:**  
- Worst-case time per command: $O(n)$  
- If there are $m$ commands, the worst-case time complexity is $O(mn)$ (assuming each command might traverse the list in the worst case).

## Space Complexity

**Auxiliary Space:** The code uses a constant amount of extra space $O(1)$ for pointer variables and loop counters.

**Data Structure Space:** The linked list uses $O(n)$ space for storing n nodes.

**Overall:** $O(n)$, where n is the number of nodes in the list.
