#ifndef CAP_H
#define CAP_H

#include <cstddef>
#include <vector>
#include <numeric>
#include <set>
#include <map>
#include <algorithm>
#include "sdp_branch_and_bound.h"

typedef struct BResult {
    double heu_mss = 0.0;
    double lb = 0.0;
    double lb_plus = 0.0;
    double avg_gap = 0.0;
    double min_gap = 0.0;
    double max_gap = 0.0;
    double avg_cl_subproblem = 0.0;
    double min_cl_subproblem = 0.0;
    double max_cl_subproblem = 0.0;
    double sum_cl_subproblem = 0.0;
    double unc_lb = 0.0;
    double unc_ub = 0.0;
    double vns_dps = -1.0;
    double vns_time = 0.0;
    double rm_mss = -1.0;
    double lb_time = 0.0;
    arma::uvec final_labels;
    bool has_final_labels = false;
    std::string status = "OK";
    std::string cl_per_subproblem;
    std::string points_per_subproblem;
} BResult;

struct ConnectedComponents {
    std::vector<std::vector<int>> comps;   // comps[c] = global point indices in component c
    std::vector<char> present;             // present[i] = true if point i appears in any CL pair
};

struct MoveUnit
{
    std::vector<int> points; // global point indices that must move together
};

struct RefinementState
{
    std::vector<int> unit_ac;        // size m, AC id in [0,t)
    std::vector<int> mounted_label;  // size n, -1 if AC not yet evaluated
    long long agreement = 0;         // # points where mounted_label == original_labels
};

template<class T>
class Matrix {
public:
    int rows;
    int cols;

private:
    std::vector<T> data;

public:
    Matrix(int row, int col) : rows(row), cols(col), data(static_cast<size_t>(row) * static_cast<size_t>(col)) {
    }

    T &operator()(size_t row, size_t col) {
        return data[row * static_cast<size_t>(cols) + col];
    }

    const T &operator()(size_t row, size_t col) const {
        return data[row * static_cast<size_t>(cols) + col];
    }
};


double compute_mss(const arma::mat &data, const arma::mat &sol);

// Run the same randomized backtracking phase used to initialize the VNS.
// This is a feasibility check only: no partition objective is optimized.
bool can_construct_vns_start(
    const std::vector<std::vector<int>> &unit_cluster_count,
    const std::vector<int> &n_lb,
    const std::vector<int> &n_ub,
    int n_anticlusters,
    int attempts = 10000,
    unsigned seed = 42);

void cap(arma::mat &Ws, UserConstraints &constraints,
double heu_mss, arma::mat &heu_sol, ConnectedComponents &cc, BResult &results);

#endif //CAP_H
