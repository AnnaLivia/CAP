#include <iostream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stack>
#include <system_error>
#include <thread>
#include <mutex>
#include <limits>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <armadillo>
#include "cap.h"
#include "config_params.h"
#include "sdp_branch_and_bound.h"
namespace fs = std::filesystem;

// data file
const char *data_path;
const char *constr_path;
const char *sol_path;

// log and result path
std::string inst_name;
std::string inst_num;
std::string inst_perc;
std::string inst_rep;
std::string result_folder;
std::string result_path;
std::ofstream log_file;
std::mutex file_mutex;
std::ofstream lb_file;

// instance data
int n;
int d;
int k;
int t;
int m;
int max_component_size;
std::string component_size_distribution;

// Automatic choice of the number of anticlusters.  These constants define the
// nominal candidate range; component indivisibility and cluster-wise balance
// are checked separately for each candidate.
constexpr int AUTO_T_MIN_SUBPROBLEM_SIZE = 100;
constexpr int AUTO_T_BASE_MAX_SUBPROBLEM_SIZE = 400;

// anticlustering
int n_threads_part;
int sdp_solver_max_concurrent_solves;

// kmeans
int kmeans_max_iter;
int kmeans_n_start;
bool kmeans_verbose;
int kmeans_permutations;

// branch and bound
double branch_and_bound_tol;
int branch_and_bound_parallel;
int branch_and_bound_max_nodes;
int branch_and_bound_visiting_strategy;

// sdp solver
int sdp_solver_session_threads_root;
int sdp_solver_session_threads;
const char *sdp_solver_folder;
double sdp_solver_tol;
int sdp_solver_stopoption;
int sdp_solver_maxiter;
int sdp_solver_maxtime;
int sdp_solver_verbose;
int sdp_solver_max_cp_iter_root;
int sdp_solver_max_cp_iter;
double sdp_solver_cp_tol;
int sdp_solver_max_ineq;
double sdp_solver_inherit_perc;
double sdp_solver_eps_ineq;
double sdp_solver_eps_active;
int sdp_solver_max_pair_ineq;
double sdp_solver_pair_perc;
int sdp_solver_max_triangle_ineq;
double sdp_solver_triangle_perc;

static std::string format_dps_value(double v) {
    std::ostringstream oss;
    const double av = std::abs(v);
    if (av > 0.0 && av < 5e-4)
        oss << std::scientific << std::setprecision(3) << v;
    else
        oss << std::fixed << std::setprecision(3) << v;
    return oss.str();
}

// read parameters in config file
std::map<std::string, std::string> read_params(const std::string &path) {
    std::map<std::string, std::string> cfg;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Reading Parameter File: Couldn't open config file: " << path << "\n";
        return cfg;
    }

    std::string line;
    while (std::getline(in, line)) {
        // trim spaces and tabs
        line.erase(std::remove_if(line.begin(), line.end(),
                                  [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r'; }),
                   line.end());
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos || pos == 0 || pos + 1 >= line.size()) continue;
        cfg.emplace(line.substr(0, pos), line.substr(pos + 1));
    }
    return cfg;
}

// read data
arma::mat read_data(const char *filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // read the header n, d, k
    file >> n >> d >> k;
    arma::mat data(n, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            file >> data(i, j);
        }
    }

    return data;
}

// read constraints
void read_constraints(const char *filename, int n, UserConstraints &constraints) {

    double delta = 0.0;
    double gamma = std::numeric_limits<double>::infinity();

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }
    constraints.ml_pairs.clear();
    constraints.cl_pairs.clear();
    constraints.cl_weights.clear();
    constraints.cl_per_point.clear();
    constraints.cl_per_point.resize(n);
    constraints.ml_per_point.clear();
    constraints.ml_per_point.resize(n);
    std::string constraint_type;

    while (file >> constraint_type) {
        if (constraint_type == "ML" || constraint_type == "CL") {
            int a, b;
            file >> a >> b;
            if (!file) {
                std::cerr << "Reading Constraints: Malformed line after " << constraint_type << "\n";
                exit(EXIT_FAILURE);
            }
            if (a < 0 || a >= n || b < 0 || b >= n) {
                std::cerr << "Reading Constraints: Invalid index in " << constraint_type << ": " << a << " " << b << "\n";
                exit(EXIT_FAILURE);
            }
            std::pair<int, int> ab_pair(a, b);
            if (constraint_type == "ML") {
                constraints.ml_pairs.push_back(ab_pair);
                constraints.ml_per_point[a].push_back(b);
                constraints.ml_per_point[b].push_back(a);
            } else {
                constraints.cl_pairs.push_back(ab_pair);
                constraints.cl_per_point[a].push_back(b);
                constraints.cl_per_point[b].push_back(a);
                constraints.cl_weights.push_back(1);
            }
        } else {
            std::cerr << "Reading Constraints: Unknown constraint type " << constraint_type << "\n";
            exit(EXIT_FAILURE);
        }
    }

}

// read solution
arma::mat read_sol(const char *filename, const UserConstraints &constraints) {
    std::ifstream f(filename);
    if (!f) {
        std::cerr << strerror(errno) << "\n";
        std::exit(EXIT_FAILURE);
    }

    // peek first line to infer columns
    std::streampos start = f.tellg();
    std::string first;
    std::getline(f, first);
    std::istringstream iss(first);
    int cols = 0;
    double tmp;
    while (iss >> tmp) ++cols;

    // rewind
    f.clear();
    f.seekg(start);

    arma::mat sol(n, k, arma::fill::zeros);

    if (cols == 1) {
        // one label per row
        for (int i = 0; i < n; ++i) {
            int c;
            f >> c;
            if (!f) {
                std::cerr << "Reading Heuristic Solution: failed while reading label for row " << i << "\n";
                std::exit(EXIT_FAILURE);
            }
            if (c < 0 || c >= k) {
                std::cerr << "Reading Heuristic Solution: label out of range at row " << i
                        << " -> " << c << "\n";
                std::exit(EXIT_FAILURE);
            }
            sol(i, c) = 1.0;
        }
    } else {
        if (cols != k) {
            std::cerr << "Reading Heuristic Solution: expected " << k
                    << " columns, got " << cols << "\n";
            std::exit(EXIT_FAILURE);
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                f >> sol(i, j);
                if (!f) {
                    std::cerr << "Reading Heuristic Solution: failed while reading entry ("
                            << i << "," << j << ")\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }

    // check that sol is a valid hard clustering matrix
    arma::uvec labels(n);
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        int lab = -1;
        for (int j = 0; j < k; ++j) {
            double v = sol(i, j);

            if (v != 0.0 && v != 1.0) {
                std::cerr << "Reading Heuristic Solution: solution is not binary at ("
                        << i << "," << j << "), value = " << v << "\n";
                std::exit(EXIT_FAILURE);
            }

            if (v == 1.0) {
                ++cnt;
                lab = j;
            }
        }

        if (cnt != 1) {
            std::cerr << "Reading Heuristic Solution: row " << i
                    << " does not contain exactly one 1\n";
            std::exit(EXIT_FAILURE);
        }

        labels(i) = lab;
    }

    // check ML constraints
    for (const auto &pr: constraints.ml_pairs) {
        int a = pr.first;
        int b = pr.second;
        if (labels(a) != labels(b)) {
            std::cerr << "Reading Heuristic Solution: violated ML constraint (" << a << ", " << b
                    << ") with labels " << labels(a) << " and " << labels(b) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    // check CL constraints
    for (const auto &pr: constraints.cl_pairs) {
        int a = pr.first;
        int b = pr.second;
        if (labels(a) == labels(b)) {
            std::cerr << "Reading Heuristic Solution: violated CL constraint (" << a << ", " << b
                    << ") both assigned to cluster " << labels(a) << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    return sol;
}

// compute connected components induced by the ML/CL-constraint graph
ConnectedComponents compute_connected_components(int n, const UserConstraints &constraints) {
    ConnectedComponents cc;
    std::vector<std::vector<int>> adj(n);
    cc.present.assign(n, false);

    for (const auto& e : constraints.cl_pairs) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        cc.present[e.first] = true;
        cc.present[e.second] = true;
    }
    for (const auto& e : constraints.ml_pairs) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        cc.present[e.first] = true;
        cc.present[e.second] = true;
    }

    std::vector<char> vis(n, false);

    for (int s = 0; s < n; ++s) {
        if (!cc.present[s] || vis[s]) continue;

        std::vector<int> comp;
        std::stack<int> st;
        st.push(s);
        vis[s] = true;

        while (!st.empty()) {
            int u = st.top(); st.pop();
            comp.push_back(u);
            for (int v : adj[u])
                if (!vis[v]) { vis[v] = true; st.push(v); }
        }
        cc.comps.push_back(std::move(comp));
    }

    return cc;
}

struct ComponentStats {
    int max_size = 1;
    std::string distribution;
};

static ComponentStats build_component_stats(const ConnectedComponents &cc, int n) {
    std::map<int, int> size_count;
    int max_size = 1;
    int present_count = 0;

    for (bool p : cc.present)
        if (p) present_count++;

    for (const auto &comp : cc.comps) {
        const int sz = static_cast<int>(comp.size());
        if (sz <= 0)
            continue;
        size_count[sz]++;
        max_size = std::max(max_size, sz);
    }

    const int singletons = n - present_count;
    if (singletons > 0)
        size_count[1] += singletons;

    std::ostringstream oss;
    bool first = true;
    for (const auto &[sz, cnt] : size_count) {
        if (!first)
            oss << ",";
        oss << sz << ":" << cnt;
        first = false;
    }
    if (first)
        oss << "1:" << n;

    return ComponentStats{max_size, oss.str()};
}

struct AutoTCandidate {
    int t = 1;
};

struct AutoTSelection {
    AutoTCandidate selected;
    int lower_t = 1;
    int upper_t = 1;
    int largest_component = 1;
    int min_nominal_size = AUTO_T_MIN_SUBPROBLEM_SIZE;
    int max_nominal_size = AUTO_T_BASE_MAX_SUBPROBLEM_SIZE;
};

static AutoTSelection compute_auto_t(const arma::mat &reference_solution,
                                     const ConnectedComponents &cc,
                                     int n,
                                     const std::string &profile) {
    if (reference_solution.n_rows == 0 || reference_solution.n_cols == 0)
        throw std::runtime_error("compute_auto_t: reference solution must be non-empty.");
    if (n <= 0)
        throw std::runtime_error("compute_auto_t: n must be positive.");

    std::vector<int> labels(reference_solution.n_rows, -1);
    std::vector<int> cluster_sizes(reference_solution.n_cols, 0);
    for (arma::uword i = 0; i < reference_solution.n_rows; ++i) {
        for (arma::uword j = 0; j < reference_solution.n_cols; ++j) {
            if (reference_solution(i, j) != 1.0)
                continue;
            labels[i] = static_cast<int>(j);
            cluster_sizes[j]++;
            break;
        }
        if (labels[i] < 0)
            throw std::runtime_error(
                "compute_auto_t: invalid reference assignment.");
    }

    for (int cluster_size : cluster_sizes) {
        if (cluster_size <= 0)
            throw std::runtime_error(
                "compute_auto_t: the reference solution contains an empty cluster.");
    }

    // Build the same units, in the same order, used by the VNS: connected
    // components first, followed by unconstrained singleton points.
    std::vector<std::vector<int>> unit_cluster_count;
    unit_cluster_count.reserve(cc.comps.size() + n);
    std::vector<char> point_in_component(n, false);
    for (const auto &component : cc.comps) {
        if (component.empty())
            continue;
        std::vector<int> counts(cluster_sizes.size(), 0);
        for (int point : component) {
            if (point < 0 || point >= n)
                throw std::runtime_error(
                    "compute_auto_t: invalid point index in connected component.");
            if (point_in_component[point])
                throw std::runtime_error(
                    "compute_auto_t: point appears in multiple connected components.");
            point_in_component[point] = true;
            counts[labels[point]]++;
        }
        unit_cluster_count.push_back(std::move(counts));
    }
    for (int point = 0; point < n; ++point) {
        if (point_in_component[point])
            continue;
        std::vector<int> counts(cluster_sizes.size(), 0);
        counts[labels[point]] = 1;
        unit_cluster_count.push_back(std::move(counts));
    }

    int largest_component = 1;
    for (const auto &comp : cc.comps)
        largest_component = std::max(largest_component,
                                     static_cast<int>(comp.size()));

    // B0 excludes nominal anticluster sizes below the minimum target or below
    // the largest connected component. B1 is the nominal upper size scale.
    // Actual sizes may exceed B1 because connected components are indivisible.
    const int min_nominal_size = std::max(
        AUTO_T_MIN_SUBPROBLEM_SIZE, largest_component);
    const int max_nominal_size = std::max(
        AUTO_T_BASE_MAX_SUBPROBLEM_SIZE,
        static_cast<int>(std::ceil(1.2 * largest_component)));
    // Round the lower endpoint down because indivisible components can force
    // actual anticlusters above the nominal B1 scale.
    const int lower_t = std::max(1, n / max_nominal_size);
    const int upper_t = std::max(1, n / min_nominal_size);

    std::vector<AutoTCandidate> feasible_candidates;
    if (lower_t <= upper_t) {
        for (int candidate_t = lower_t; candidate_t <= upper_t; ++candidate_t) {
            std::vector<int> lower(cluster_sizes.size(), 0);
            std::vector<int> upper(cluster_sizes.size(), 0);
            for (size_t j = 0; j < cluster_sizes.size(); ++j) {
                const int per_anticluster = cluster_sizes[j] / candidate_t;
                lower[j] = static_cast<int>(0.8 * per_anticluster);
                upper[j] = static_cast<int>(1.2 * per_anticluster);
            }

            bool compatible = true;
            for (const auto &component_counts : unit_cluster_count) {
                for (size_t j = 0; j < component_counts.size(); ++j) {
                    if (component_counts[j] > upper[j]) {
                        compatible = false;
                        break;
                    }
                }
                if (!compatible)
                    break;
            }

            // Test the complete packing with the exact same randomized
            // backtracking constructor later used by the VNS itself.
            if (compatible && can_construct_vns_start(
                    unit_cluster_count, lower, upper, candidate_t)) {
                feasible_candidates.push_back({candidate_t});
            }
        }
    }

    if (feasible_candidates.empty()) {
        throw std::runtime_error(
            "compute_auto_t: no T satisfies the nominal size range and a "
            "VNS-feasible component packing.");
    }

    const AutoTCandidate &min_candidate = feasible_candidates.front();
    const AutoTCandidate &max_candidate = feasible_candidates.back();
    const double mid_target =
        (static_cast<double>(min_candidate.t) + max_candidate.t) / 2.0;
    const auto mid_it = std::min_element(
        feasible_candidates.begin(), feasible_candidates.end(),
        [mid_target](const AutoTCandidate &lhs, const AutoTCandidate &rhs) {
            return std::abs(lhs.t - mid_target) <
                   std::abs(rhs.t - mid_target);
        });

    AutoTCandidate selected;
    if (profile == "min")
        selected = min_candidate;
    else if (profile == "max")
        selected = max_candidate;
    else
        selected = *mid_it;

    return {selected, min_candidate.t, max_candidate.t, largest_component,
            min_nominal_size, max_nominal_size};
}

static void save_improved_labels_if_needed(const arma::uvec &labels,
                                           double init_ub,
                                           double final_ub) {
    if (!std::isfinite(final_ub) || !(final_ub < init_ub))
        return;
    if (labels.n_elem != static_cast<arma::uword>(n))
        return;

    fs::path labels_dir = fs::path(result_folder) / "labels";
    fs::create_directories(labels_dir);

    const std::string base_name = fs::path(result_path).filename().string();
    fs::path out_path = labels_dir / (base_name + "_improved.txt");

    std::ofstream out(out_path.string());
    if (!out)
        throw std::runtime_error(
            "save_improved_labels_if_needed: cannot open " + out_path.string());

    for (arma::uword i = 0; i < labels.n_elem; ++i)
        out << labels(i) << "\n";

    std::cout << "Saved improved labels to: " << out_path.string() << "\n";
    log_file << "Saved improved labels to: " << out_path.string() << "\n";
}

// write log preamble for instance
void write_log_preamble(double heu_mss) {

    std::string str_path = data_path;
    log_file << "INSTANCE, N, D, K, T, M:\n";
    log_file << str_path.substr(str_path.find_last_of("/\\") + 1) << " " << n << " " << d << " " << k << " " << t << " " << m << "\n\n";
    log_file << "PARTITION_METHOD: VNS\n";
    log_file << "MAX_COMPONENT_SIZE: " << max_component_size << "\n";
    log_file << "COMPONENT_SIZE_DISTRIBUTION: " << component_size_distribution << "\n\n";

    log_file << "PARTITION_THREADS: " << n_threads_part << "\n\n";
    log_file << "SDP_SOLVER_MAX_CONCURRENT_SOLVES: " << sdp_solver_max_concurrent_solves << "\n\n";
    log_file << "KMEANS_NUM_START: " << kmeans_n_start << "\n";
    log_file << "KMEANS_MAX_ITERATION: " << kmeans_max_iter << "\n";
    log_file << "KMEANS_NUM_PERMUTATIONS: " << kmeans_permutations << "\n";
    log_file << "KMEANS_VERBOSE: " << kmeans_verbose << "\n\n";

    log_file << "BRANCH_AND_BOUND_TOL: " << branch_and_bound_tol << "\n";
    log_file << "BRANCH_AND_BOUND_PARALLEL: " << branch_and_bound_parallel << "\n";
    log_file << "BRANCH_AND_BOUND_MAX_NODES: " << branch_and_bound_max_nodes << "\n";
    log_file << "BRANCH_AND_BOUND_VISITING_STRATEGY: " << branch_and_bound_visiting_strategy << "\n\n";

    log_file << "SDP_SOLVER_SESSION_THREADS_ROOT: " << sdp_solver_session_threads_root << "\n";
    log_file << "SDP_SOLVER_SESSION_THREADS: " << sdp_solver_session_threads << "\n";
    log_file << "SDP_SOLVER_FOLDER: " << sdp_solver_folder << "\n";
    log_file << "SDP_SOLVER_TOL: " << sdp_solver_tol << "\n";
    log_file << "SDP_SOLVER_VERBOSE: " << sdp_solver_verbose << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER_ROOT: " << sdp_solver_max_cp_iter_root << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER: " << sdp_solver_max_cp_iter << "\n";
    log_file << "SDP_SOLVER_CP_TOL: " << sdp_solver_cp_tol << "\n";
    log_file << "SDP_SOLVER_MAX_INEQ: " << sdp_solver_max_ineq << "\n";
    log_file << "SDP_SOLVER_INHERIT_PERC: " << sdp_solver_inherit_perc << "\n";
    log_file << "SDP_SOLVER_EPS_INEQ: " << sdp_solver_eps_ineq << "\n";
    log_file << "SDP_SOLVER_EPS_ACTIVE: " << sdp_solver_eps_active << "\n";
    log_file << "SDP_SOLVER_MAX_PAIR_INEQ: " << sdp_solver_max_pair_ineq << "\n";
    log_file << "SDP_SOLVER_PAIR_PERC: " << sdp_solver_pair_perc << "\n";
    log_file << "SDP_SOLVER_MAX_TRIANGLE_INEQ: " << sdp_solver_max_triangle_ineq << "\n";
    log_file << "SDP_SOLVER_TRIANGLE_PERC: " << sdp_solver_triangle_perc << "\n\n";
    log_file << std::fixed << std::setprecision(3);

    log_file << "\n---------------------------------------------------------------------\n";
    log_file << "UB: " << heu_mss << "\n";
}

void run(int argc, char **argv) {
    std::string config_file = "config.txt";
    std::map<std::string, std::string> config_map = read_params(config_file);

    result_folder = config_map["RESULT_FOLDER"];
    std::error_code output_directory_error;
    fs::create_directories(result_folder, output_directory_error);
    if (output_directory_error) {
        throw std::runtime_error(
            "Cannot create result directory '" + result_folder + "': "
            + output_directory_error.message());
    }

    // number of thread for computing the partition bound
    n_threads_part = std::stoi(config_map["PARTITION_THREADS"]);
    {
        auto it = config_map.find("SDP_SOLVER_MAX_CONCURRENT_SOLVES");
        if (it != config_map.end() && !it->second.empty())
            sdp_solver_max_concurrent_solves = std::stoi(it->second);
        else
            sdp_solver_max_concurrent_solves = 2;
    }
    // kmeans
    kmeans_max_iter = std::stoi(config_map["KMEANS_MAX_ITERATION"]);
    kmeans_n_start = std::stoi(config_map["KMEANS_NUM_START"]);
    kmeans_permutations = std::stoi(config_map["KMEANS_NUM_PERMUTATIONS"]);
    kmeans_verbose = std::stoi(config_map["KMEANS_VERBOSE"]);

    // branch and bound
    branch_and_bound_tol = std::stod(config_map["BRANCH_AND_BOUND_TOL"]);
    branch_and_bound_parallel = std::stoi(config_map["BRANCH_AND_BOUND_PARALLEL"]);
    branch_and_bound_max_nodes = std::stoi(config_map["BRANCH_AND_BOUND_MAX_NODES"]);
    branch_and_bound_visiting_strategy = std::stoi(config_map["BRANCH_AND_BOUND_VISITING_STRATEGY"]);

    // sdp solver
    sdp_solver_session_threads_root = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS_ROOT"]);
    sdp_solver_session_threads = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS"]);
    sdp_solver_folder = config_map["SDP_SOLVER_FOLDER"].c_str();
    sdp_solver_tol = std::stod(config_map["SDP_SOLVER_TOL"]);
    sdp_solver_verbose = std::stoi(config_map["SDP_SOLVER_VERBOSE"]);
    sdp_solver_max_cp_iter_root = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER_ROOT"]);
    sdp_solver_max_cp_iter = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER"]);
    sdp_solver_cp_tol = std::stod(config_map["SDP_SOLVER_CP_TOL"]);
    sdp_solver_max_ineq = std::stoi(config_map["SDP_SOLVER_MAX_INEQ"]);
    sdp_solver_inherit_perc = std::stod(config_map["SDP_SOLVER_INHERIT_PERC"]);
    sdp_solver_eps_ineq = std::stod(config_map["SDP_SOLVER_EPS_INEQ"]);
    sdp_solver_eps_active = std::stod(config_map["SDP_SOLVER_EPS_ACTIVE"]);
    sdp_solver_max_pair_ineq = std::stoi(config_map["SDP_SOLVER_MAX_PAIR_INEQ"]);
    sdp_solver_pair_perc = std::stod(config_map["SDP_SOLVER_PAIR_PERC"]);
    sdp_solver_max_triangle_ineq = std::stoi(config_map["SDP_SOLVER_MAX_TRIANGLE_INEQ"]);
    sdp_solver_triangle_perc = std::stod(config_map["SDP_SOLVER_TRIANGLE_PERC"]);
    sdp_solver_stopoption = 0;
    sdp_solver_maxiter = 50000;
    sdp_solver_maxtime = 3 * 3600;

    if (argc < 4 || argc > 6) {
        std::cerr << "Input: <DATA_FILE> <CONSTRAINT_FILE> <SOL_FILE> "
                     "[T_OR_AUTO] [AUTO_T_PROFILE]\n";
        std::cerr << "T_OR_AUTO: empty|auto => auto-computed, or a positive integer\n";
        std::cerr << "AUTO_T_PROFILE: min|mid|max (default: mid; applies only to auto)\n";
        exit(EXIT_FAILURE);
    }

    // read input parameters
    data_path = argv[1];
    constr_path = argv[2];
    sol_path = argv[3];
    int t_override = -1;
    std::string auto_t_profile = "mid";
    bool seen_t_option = false;
    bool seen_auto_t_profile = false;
    auto parse_t_override = [&](const std::string &tok) {
        if (tok.empty() || tok == "auto") {
            seen_t_option = true;
            return;
        }
        try {
            int parsed = std::stoi(tok);
            if (parsed <= 0)
                throw std::runtime_error("non-positive");
            t_override = parsed;
            seen_t_option = true;
        } catch (...) {
            std::cerr << "Invalid T_OR_AUTO '" << tok << "'. Use empty|auto or a positive integer.\n";
            exit(EXIT_FAILURE);
        }
    };
    for (int arg_idx = 4; arg_idx < argc; ++arg_idx) {
        const std::string token = argv[arg_idx];
        if (token == "min" || token == "mid" || token == "max") {
            if (seen_auto_t_profile) {
                std::cerr << "AUTO_T_PROFILE was specified more than once.\n";
                exit(EXIT_FAILURE);
            }
            auto_t_profile = token;
            seen_auto_t_profile = true;
        } else {
            if (seen_t_option) {
                std::cerr << "T_OR_AUTO was specified more than once.\n";
                exit(EXIT_FAILURE);
            }
            parse_t_override(token);
        }
    }

    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    n_threads_part = std::clamp(n_threads_part, 1, (int) hw);
    sdp_solver_max_concurrent_solves = std::clamp(
        sdp_solver_max_concurrent_solves, 1, n_threads_part);

    arma::mat data = read_data(data_path);  // sets n, d, k from file header

    std::string str_path = data_path;
    inst_name = str_path.substr(str_path.find_last_of("/\\") + 1);
    inst_name = inst_name.substr(0, inst_name.find("."));

    std::string inst_full_name = constr_path;
    inst_full_name = inst_full_name.substr(inst_full_name.find_last_of("/\\") + 1);
    inst_full_name = inst_full_name.substr(0, inst_full_name.find("_cl.txt"));
    result_path = result_folder + "/" + inst_full_name;

    BResult results;

    // printing instance info
    std::cout << "---------------------------------------------------------------------\n";
    std::cout << "Instance " << inst_full_name << "\n";
    std::cout << "Num Clusters " << k << "\n";
    std::cout << "Num Points " << n << "\n";
    std::cout << "Num Features " << d << "\n";
    std::cout << "Partition Method VNS\n";

    // read constraints
    UserConstraints constraints;
    read_constraints(constr_path, n, constraints);
    int ml = constraints.ml_pairs.size();
    int cl = constraints.cl_pairs.size();

    // printing instance info
    if (ml > 0)
        std::cout << "Num ML Constraints " << ml << "\n";
    if (cl > 0)
        std::cout << "Num CL Constraints " << cl << "\n";

    // create connected components
    ConnectedComponents cc = compute_connected_components(n, constraints);

    // Read the reference partition before setting T: its cluster sizes and the
    // connected-component structure define the automatic feasible range.
    arma::mat heu_sol = read_sol(sol_path, constraints);
    AutoTSelection auto_t_selection;
    if (t_override > 0) {
        t = t_override;
    } else {
        auto_t_selection = compute_auto_t(
            heu_sol, cc, n, auto_t_profile);
        t = auto_t_selection.selected.t;
    }
    int present_count = 0;
    for (char p : cc.present)
        if (p) present_count++;
    const int singleton_components = n - present_count;
    m = static_cast<int>(cc.comps.size()) + singleton_components;
    const ComponentStats component_stats = build_component_stats(cc, n);
    max_component_size = component_stats.max_size;
    component_size_distribution = component_stats.distribution;

    // printing instance info
    std::cout << "Num Connected Components " << m << "\n";
    std::cout << "Max Component Size " << max_component_size << "\n";
    std::cout << "Component Size Distribution " << component_size_distribution << "\n";
    if (t_override > 0) {
        std::cout << "Num Anticlusters (manual) " << t << "\n";
    } else {
        std::cout << "Num Anticlusters (auto) " << t
                  << " (profile " << auto_t_profile << ", range "
                  << auto_t_selection.lower_t << "--" << auto_t_selection.upper_t
                  << ", nominal size range "
                  << auto_t_selection.min_nominal_size << "--"
                  << auto_t_selection.max_nominal_size << ")\n";
        std::cout << "Auto-T nominal subproblem size range "
                  << auto_t_selection.min_nominal_size << "--"
                  << auto_t_selection.max_nominal_size
                  << " (largest component " << auto_t_selection.largest_component
                  << ")\n";
    }

    results.heu_mss = compute_mss(data, heu_sol);

    // printing instance info
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nHeuristic MSS: " << results.heu_mss << "\n";
    std::cout << "---------------------------------------------------------------------\n\n";

    log_file.open(result_path + "_T" + std::to_string(t) + "_VNS_LOG.txt");
    write_log_preamble(results.heu_mss);
    if (t_override <= 0) {
        log_file << "AUTO_T_PROFILE: " << auto_t_profile << "\n";
        log_file << "AUTO_T_CANDIDATE_RANGE: " << auto_t_selection.lower_t
                 << "--" << auto_t_selection.upper_t << "\n";
        log_file << "AUTO_T_B0: " << auto_t_selection.min_nominal_size << "\n";
        log_file << "AUTO_T_B1: " << auto_t_selection.max_nominal_size << "\n";
        log_file << "AUTO_T_LARGEST_COMPONENT: "
                 << auto_t_selection.largest_component << "\n\n";
    }

    // instance_num_perc_rep, retrieve values from the instance name suffix
    // (right-to-left): ..._<inst_num>_<inst_perc>_<inst_rep>
    {
        std::vector<std::string> parts;
        std::stringstream ss(inst_full_name);
        std::string part;
        while (std::getline(ss, part, '_')) {
            if (!part.empty())
                parts.push_back(part);
        }

        inst_num = parts[parts.size() - 3];
        inst_perc = parts[parts.size() - 2];
        inst_rep = parts[parts.size() - 1];
    }

    const std::string test_summary_path = result_folder + "/test_SUMMARY.txt";
    const bool write_summary_header =
        !fs::exists(test_summary_path) || fs::is_empty(test_summary_path);
    std::ofstream test_SUMMARY(test_summary_path, std::ios::app);

    if (write_summary_header) {
        test_SUMMARY
            << "inst_name\tinst_num\tinst_perc\tinst_rep\t"
            << "n\td\tcl\tk\tt\t"
            << "connected_components\tmax_component_size\tcomponent_size_distribution\t"
            << "cl_total\tcl_represented_sum\tcl_represented_pct\t"
            << "avg_cl_per_subproblem\tmin_cl_per_subproblem\tmax_cl_per_subproblem\t"
            << "cl_per_subproblem\tpoints_per_subproblem\t"
            << "heu_mss\tfinal_ub\timpr_ub_pct\tvns_dps\t"
            << "lb\tlb_plus\t"
            << "gap_lb_pct\tgap_lbplus_pct\tavg_gap_pcsdp_pct\tmin_gap_pcsdp_pct\tmax_gap_pcsdp_pct\t"
            << "vns_time_s\tlb_time_sec\ttotal_min\trun_status\n";
    }

    test_SUMMARY << std::fixed << std::setprecision(3)
            << inst_name << "\t"
            << inst_num << "\t"
            << inst_perc << "\t"
            << inst_rep << "\t"
            << n << "\t"
            << d << "\t"
            << cl << "\t"
            << k << "\t"
            << t << "\t"
            << m << "\t"
            << max_component_size << "\t"
            << component_size_distribution << "\t";

    test_SUMMARY << std::fixed << std::setprecision(3);

    auto write_summary_na_row = [&test_SUMMARY]() {
        constexpr int summary_columns = 23;
        for (int i = 0; i < summary_columns - 1; ++i)
            test_SUMMARY << "NA\t";
        test_SUMMARY << "ERR\n";
    };

    // Run anticlustering and compute bounds
    try {
        cap(data, constraints, results.heu_mss, heu_sol, cc, results);
        double final_mss = (results.heu_mss < results.rm_mss) and (results.rm_mss > 0) ? results.heu_mss : results.rm_mss;
        const double cl_total = static_cast<double>(cl);
        const double cl_represented = results.sum_cl_subproblem;
        const double cl_represented_pct = (cl_total > 0.5)
            ? (100.0 * cl_represented / cl_total)
            : 0.0;

        if (results.has_final_labels)
            save_improved_labels_if_needed(results.final_labels, results.heu_mss, final_mss);

        std::cout << "\n------------------------------------------------------------\n";
        std::cout << "\n[Instance stats]";
        std::cout << "\nConnected components: " << m;
        std::cout << "\nMax component size: " << max_component_size;
        std::cout << "\nComponent size distribution: " << component_size_distribution;
        std::cout << "\nCL_TOTAL: " << cl_total;
        std::cout << "\nCL_REPRESENTED_SUM: " << cl_represented;
        std::cout << "\nCL_REPRESENTED(%): " << cl_represented_pct;
        std::cout << "\nAVG_CL_PER_SUBPROBLEM: " << results.avg_cl_subproblem;
        std::cout << "\nMIN_CL_PER_SUBPROBLEM: " << results.min_cl_subproblem;
        std::cout << "\nMAX_CL_PER_SUBPROBLEM: " << results.max_cl_subproblem;
        std::cout << "\nCL_PER_SUBPROBLEM: " << results.cl_per_subproblem;
        std::cout << "\nPOINTS_PER_SUBPROBLEM: " << results.points_per_subproblem;
        std::cout << "\n\n[VNS]";
        std::cout << "\nUB: " << results.heu_mss;
        std::cout << "\nFinal UB: " << results.rm_mss;
        if (results.vns_dps >= 0.0)
            std::cout << "\nVNS_D(P,S): " << format_dps_value(results.vns_dps);
        std::cout << "\n\n[LB]";
        std::cout << "\nLB: " << results.lb;
        std::cout << "\nLB+: " << results.lb_plus;
        std::cout << "\n\n[Gap and times]";
        std::cout << "\nGAP_LB(%): " << (final_mss - results.lb) * 100 / final_mss;
        std::cout << "\nGAP_LB+(%): " << (final_mss - results.lb_plus) * 100 / final_mss;
        std::cout << "\nAVG_GAP_PCSDP(%): " << (100.0 * results.avg_gap);
        std::cout << "\nMIN_GAP_PCSDP(%): " << (100.0 * results.min_gap);
        std::cout << "\nMAX_GAP_PCSDP(%): " << (100.0 * results.max_gap);
        std::cout << "\nVNS_TIME(s): " << results.vns_time;
        std::cout << "\nLB_TIME(s): " << results.lb_time;
        std::cout << "\nTOTAL_MIN: " << ((results.vns_time + results.lb_time) / 60);
        std::cout << "\n------------------------------------------------------------\n\n";

        log_file << "\n------------------------------------------------------------\n";
        log_file << "\n[Instance stats]";
        log_file << "\nConnected components: " << m;
        log_file << "\nMax component size: " << max_component_size;
        log_file << "\nComponent size distribution: " << component_size_distribution;
        log_file << "\nCL_TOTAL: " << cl_total;
        log_file << "\nCL_REPRESENTED_SUM: " << cl_represented;
        log_file << "\nCL_REPRESENTED(%): " << cl_represented_pct;
        log_file << "\nAVG_CL_PER_SUBPROBLEM: " << results.avg_cl_subproblem;
        log_file << "\nMIN_CL_PER_SUBPROBLEM: " << results.min_cl_subproblem;
        log_file << "\nMAX_CL_PER_SUBPROBLEM: " << results.max_cl_subproblem;
        log_file << "\nCL_PER_SUBPROBLEM: " << results.cl_per_subproblem;
        log_file << "\nPOINTS_PER_SUBPROBLEM: " << results.points_per_subproblem;
        log_file << "\n\n[VNS]";
        log_file << "\nUB: " << results.heu_mss;
        log_file << "\nFinal UB: " << results.rm_mss;
        if (results.vns_dps >= 0.0)
            log_file << "\nVNS_D(P,S): " << format_dps_value(results.vns_dps);
        log_file << "\n\n[LB]";
        log_file << "\nLB: " << results.lb;
        log_file << "\nLB+: " << results.lb_plus;
        log_file << "\n\n[Gap and times]";
        log_file << "\nGAP_LB(%): " << (final_mss - results.lb) * 100 / final_mss;
        log_file << "\nGAP_LB+(%): " << (final_mss - results.lb_plus) * 100 / final_mss;
        log_file << "\nAVG_GAP_PCSDP(%): " << (100.0 * results.avg_gap);
        log_file << "\nMIN_GAP_PCSDP(%): " << (100.0 * results.min_gap);
        log_file << "\nMAX_GAP_PCSDP(%): " << (100.0 * results.max_gap);
        log_file << "\nVNS_TIME(s): " << results.vns_time;
        log_file << "\nLB_TIME(s): " << results.lb_time;
        log_file << "\nTOTAL_MIN: " << ((results.vns_time + results.lb_time) / 60);
        log_file << "\n------------------------------------------------------------\n";

        test_SUMMARY << cl_total << "\t"
                     << cl_represented << "\t"
                     << cl_represented_pct << "%\t"
                     << results.avg_cl_subproblem << "\t"
                     << results.min_cl_subproblem << "\t"
                     << results.max_cl_subproblem << "\t"
                     << results.cl_per_subproblem << "\t"
                     << results.points_per_subproblem << "\t"
                     << results.heu_mss << "\t"
                     << results.rm_mss << "\t"
                     << (results.heu_mss - results.rm_mss) * 100.0 / results.heu_mss << "%\t";
        if (results.vns_dps >= 0.0)
            test_SUMMARY << format_dps_value(results.vns_dps) << "\t";
        else
            test_SUMMARY << "NA\t";
        test_SUMMARY
                << results.lb << "\t"
                << results.lb_plus << "\t"
                << (final_mss - results.lb) * 100.0 / final_mss << "%\t"
                << (final_mss - results.lb_plus) * 100.0 / final_mss << "%\t"
                << (100.0 * results.avg_gap) << "%\t"
                << (100.0 * results.min_gap) << "%\t"
                << (100.0 * results.max_gap) << "%\t"
                << results.vns_time << "\t"
                << results.lb_time << "\t"
                << (results.vns_time + results.lb_time) / 60 << "\t"
                << results.status << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Run failed: " << e.what() << "\n";
        if (log_file.is_open()) {
            log_file << "\nRun failed: " << e.what() << "\n";
            log_file.flush();
        }
        write_summary_na_row();
    } catch (...) {
        std::cerr << "Run failed: unknown exception\n";
        if (log_file.is_open()) {
            log_file << "\nRun failed: unknown exception\n";
            log_file.flush();
        }
        write_summary_na_row();
    }

    log_file.close();
    test_SUMMARY.close();
}

int main(int argc, char **argv) {
    run(argc, argv);

    return EXIT_SUCCESS;
}
