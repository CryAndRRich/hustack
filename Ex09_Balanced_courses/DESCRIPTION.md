## Description
At the beginning of the semester, the head of a computer science department $D$ has to assign courses to teachers in a balanced way. The department $D$ has:
- $m$ teachers: $T = {1, 2, ..., m}$
- $n$ courses: $C = {1, 2, ..., n}$

Each teacher $t$ ∈ T has a preference list, which is a list of courses he/she can teach depending on his/her specialization. In addition, there is a list of pairs of conflicting courses that cannot be assigned to the same teacher because they have already been scheduled in the same slot of the timetable.

The $load$ of a teacher is defined as the number of courses assigned to him/her.

The goal is to assign $n$ courses to $m$ teachers such that:
- Each course assigned to a teacher is in his/her preference list.
- No two conflicting courses are assigned to the same teacher.
- The maximum load (i.e., the maximum number of courses assigned to any teacher) is minimized.

**Input:**
- **Line 1:** Contains two integers $m$ and $n$ ($1 \leq m \leq 10, 1 \leq n \leq 30$).
- **Next $m$ lines:** Each line corresponds to a teacher. For teacher $i$ $(i = 1, …, m)$, the line begins with a positive integer $k$ (the number of courses teacher $i$ can teach), followed by $k$ positive integers indicating the courses that teacher $i$ can teach.
- **Line $m+2$:** Contains an integer $k$ indicating the number of pairs of conflicting courses.
- **Next $k$ lines:** Each of these lines contains two integers $i$ and $j$ indicating that courses $i$ and $j$ conflict and thus cannot be assigned to the same teacher.

**Output:**
- The output is a single integer which is the minimal possible value of the maximal load among all teachers in a valid assignment. If no valid assignment exists, output $-1$.

## Converting to a LP problem:
The problem is formulated as an optimization task where we need to balance the course load among teachers while satisfying the following constraints:
- **Preference Constraint:** A teacher can only be assigned courses from his/her preference list.
- **Assignment Constraint:** Every course must be assigned to exactly one teacher.
- **Conflict Constraint:** Two conflicting courses cannot be assigned to the same teacher.
- **Load Minimization:** We want to minimize the maximum number of courses (load) assigned to any teacher.

To solve this problem, we reduce it to a **Linear Programming (LP)** optimization model. However, because some variables are binary (i.e., whether a teacher is assigned a course or not), we model it as a **Mixed Integer Programming (MIP)** or **Constraint Programming (CP)** problem. Here is the step-by-step formulation:

### Decision Variables

- $x(i, j)$: A binary variable where:
  - $x(i, j) = 1$ if teacher $i$ is assigned course $j$.
  - $x(i, j) = 0$ otherwise.
  
- **L:** An integer variable representing the maximum load among all teachers.

### Constraints

- **Teacher Preference:**
    - For each teacher $i$ and course $j$ not in teacher $i$'s preference list, set:
        $$x(i, j) = 0$$
   
- **Course Assignment:**
    - For each course $j$, ensure that it is assigned to exactly one teacher:
        $$\sum_{i=1}^{m} x(i, j) = 1$$
   
- **Conflict Constraint:**
    - For each teacher $i$ and each pair of conflicting courses $(a, b)$, ensure that both courses are not assigned to the same teacher:
        $$x(i, a) + x(i, b) \leq 1$$
   
- **Load Constraint:**
    - For each teacher $i$, the total number of courses assigned should not exceed the load $L$:
        $$\sum_{j=1}^{n} x(i, j) \leq L$$
   
- **Minimize $L$:** The goal is to minimize the maximum load $L$ across all teachers

## Approach 1: Using OR-Tools
We will import and use OR-Tools, an open source software suite for optimization. We will solve the problem as a a **Mixed Integer Programming (MIP)** or **Constraint Programming (CP)** problem.

## Approach 2: Backtrack + Branch and Bound
We employ a backtracking approach combined with branch-and-bound to solve the course assignment problem. The algorithm recursively explores all possible assignments of courses to teachers and uses a heuristic to compute a lower bound on the maximum load (i.e., the number of courses assigned to any teacher). If the current load—augmented by the heuristic estimate—exceeds the best solution found so far, the algorithm prunes that branch, avoiding unnecessary computations.

### Time Complexity:
Worst-case $O(n!)$ because, in the worst-case scenario, every permutation of course assignments might be explored.

### Space Complexity:
The recursion stack depth is at most $n$, resulting in a space complexity of $O(n)$.

## Approach 3: DP + Bitmask (Failed)
This approach uses dynamic programming with bitmasking to solve the course assignment problem. Each state in the DP is represented by a bitmask that indicates which courses have been assigned so far. For each teacher, we precompute all valid subsets of courses (from his/her preference list) that respect the conflict constraints and do not exceed a given load threshold.

### Time Complexity:
The overall time complexity can be approximated as $O(m \cdot 2^n)$ for each load value checked.

### Space Complexity:
The DP table maintains $2^n$ states, leading to a space complexity of $O(2^n)$.
