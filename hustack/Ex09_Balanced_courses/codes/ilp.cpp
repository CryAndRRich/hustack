#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "ortools/linear_solver/linear_solver.h"

using namespace std;
using namespace operations_research;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    cin >> m >> n;

    vector<vector<int>> P(m + 1);
    for (int i = 1; i <= m; i++) {
        int count;
        cin >> count;
        for (int k = 0; k < count; k++) {
            int task;
            cin >> task;
            if (task >= 1 && task <= n) {
                P[i].push_back(task);
            }
        }
    }

    int k;
    cin >> k;
    vector<pair<int, int>> B;
    for (int i = 0; i < k; i++) {
        int a, b;
        cin >> a >> b;
        B.push_back({a, b});
    }

    MPSolver solver("AssignmentSolver", MPSolver::CBC_MIXED_INTEGER_PROGRAMMING);

    vector<vector<MPVariable*>> x(m + 1, vector<MPVariable*>(n + 1, nullptr));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            string var_name = "x(" + to_string(i) + "," + to_string(j) + ")";
            x[i][j] = solver.MakeIntVar(0, 1, var_name);
        }
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            bool possible = false;
            for (int task : P[i]) {
                if (task == j) {
                    possible = true;
                    break;
                }
            }
            if (!possible) {
                MPConstraint* ct = solver.MakeRowConstraint(0, 0);
                ct -> SetCoefficient(x[i][j], 1);
            }
        }
    }

    for (int j = 1; j <= n; j++) {
        MPConstraint* ct = solver.MakeRowConstraint(1, 1);
        for (int i = 1; i <= m; i++) {
            ct -> SetCoefficient(x[i][j], 1);
        }
    }

    for (int i = 1; i <= m; i++) {
        for (auto conflict : B) {
            int a = conflict.first;
            int b = conflict.second;
            MPConstraint* ct = solver.MakeRowConstraint(-solver.infinity(), 1);
            ct -> SetCoefficient(x[i][a], 1);
            ct -> SetCoefficient(x[i][b], 1);
        }
    }

    MPVariable* L = solver.MakeIntVar(0, n, "L");
    for (int i = 1; i <= m; i++) {
        MPConstraint* ct = solver.MakeRowConstraint(-solver.infinity(), 0);
        for (int j = 1; j <= n; j++) {
            ct -> SetCoefficient(x[i][j], 1);
        }
        ct -> SetCoefficient(L, -1);
    }

    MPObjective* objective = solver.MutableObjective();
    objective -> SetCoefficient(L, 1);
    objective -> SetMinimization();

    MPSolver::ResultStatus result_status = solver.Solve();

    if (result_status == MPSolver::OPTIMAL) {
        cout << int(L -> solution_value()) << "\n";
    } else {
        cout << -1 << "\n";
    }

    return 0;
}
