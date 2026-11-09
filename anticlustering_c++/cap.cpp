#include "cap.h"
#include "config_params.h"
#include "ThreadPoolPartition.h"
#include "sdp_branch_and_bound.h"
#include <algorithm>
#include <armadillo>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

static std::string format_dps_value(double v) {
    std::ostringstream oss;
    const double av = std::abs(v);
    if (av > 0.0 && av < 5e-4)
        oss << std::scientific << std::setprecision(3) << v;
    else
        oss << std::fixed << std::setprecision(3) << v;
    return oss.str();
}

// Save data to file
void save_to_file(const arma::mat &X, const std::string &name) {
    std::ofstream f(result_path + "_" + name + ".txt");
    if (!f) {
        std::cerr << "Error saving the solution to a file: cannot open\n";
        return;
    }

    // use general format (like input), not fixed
    f << std::setprecision(12);
    for (arma::uword i = 0; i < X.n_rows; ++i) {
        for (arma::uword j = 0; j < X.n_cols; ++j) {
            double v = X(i, j);
            // snap very small numerical noise
            if (std::abs(v) < 1e-12)
                v = 0.0;
            // round to reasonable precision
            v = std::round(v * 1e6) / 1e6;
            f << v;
            if (j + 1 < X.n_cols)
                f << " ";
        }
        f << "\n";
    }
}

// compute mss for a given solution
double compute_mss(const arma::mat &data, const arma::mat &sol) {
    arma::vec counts = arma::sum(sol, 0).t();
    if (arma::any(counts <= 1e-12)) {
        std::cerr << "Computing MSSC for Heuristic Solution: empty cluster found "
                "in solution matrix\n";
        exit(EXIT_FAILURE);
    }

    // centroids
    arma::mat centroids = sol.t() * data;
    centroids.each_col() /= counts;

    // labels
    arma::uvec labels = arma::index_max(sol, 1);

    // squared distances to assigned centroids
    arma::mat picked = centroids.rows(labels);
    arma::mat diff = data - picked;
    arma::vec row_sq = arma::sum(arma::square(diff), 1);

    return arma::accu(row_sq);
}

// helper
UserConstraints build_local_constraints(const arma::uvec &idx,
                                        const UserConstraints &global) {
    UserConstraints local;
    local.delta = 0.0;
    local.gamma = std::numeric_limits<double>::infinity();

    std::unordered_map<int, int> map;

    for (arma::uword i = 0; i < idx.n_elem; i++)
        map[idx(i)] = i;

    for (auto &e: global.cl_pairs) {
        auto a = map.find(e.first);
        auto b = map.find(e.second);

        if (a != map.end() && b != map.end())
            local.cl_pairs.emplace_back(a->second, b->second);
    }

    for (auto &e: global.ml_pairs) {
        auto a = map.find(e.first);
        auto b = map.find(e.second);

        if (a != map.end() && b != map.end())
            local.ml_pairs.emplace_back(a->second, b->second);
    }

    return local;
}

static std::vector<int> count_cl_per_subproblem(
    const std::vector<std::vector<int>> &ac_sol,
    const UserConstraints &constraints)
{
    std::vector<int> point_to_ac(n, -1);
    for (size_t h = 0; h < ac_sol.size(); ++h) {
        for (int p : ac_sol[h]) {
            if (p < 0 || p >= n)
                throw std::runtime_error("count_cl_per_subproblem: invalid point index.");
            if (point_to_ac[p] != -1)
                throw std::runtime_error("count_cl_per_subproblem: point assigned to multiple anticlusters.");
            point_to_ac[p] = static_cast<int>(h);
        }
    }

    for (int p = 0; p < n; ++p) {
        if (point_to_ac[p] == -1)
            throw std::runtime_error("count_cl_per_subproblem: point missing from anticluster assignment.");
    }

    std::vector<int> counts(ac_sol.size(), 0);
    for (const auto &e : constraints.cl_pairs) {
        const int a = e.first;
        const int b = e.second;
        if (a < 0 || a >= n || b < 0 || b >= n)
            throw std::runtime_error("count_cl_per_subproblem: invalid CL endpoint.");
        const int ha = point_to_ac[a];
        const int hb = point_to_ac[b];
        if (ha == hb)
            counts[ha]++;
    }

    return counts;
}

static std::vector<int> map_local_to_global_clusters(const arma::mat &local_centers,
                                                     const arma::mat &global_centers,
                                                     const std::string &ctx) {
    int local_k = static_cast<int>(local_centers.n_rows);
    int global_k = static_cast<int>(global_centers.n_rows);
    if (local_k != global_k)
        throw std::runtime_error(ctx + ": local/global cluster count mismatch");
    if (local_centers.n_cols != global_centers.n_cols)
        throw std::runtime_error(ctx + ": local/global feature dimension mismatch");

    arma::vec n1 = arma::sum(arma::square(local_centers), 1);
    arma::vec n2 = arma::sum(arma::square(global_centers), 1);
    arma::mat D = n1 * arma::ones<arma::rowvec>(global_k) +
                  arma::ones<arma::vec>(local_k) * n2.t() -
                  2.0 * (local_centers * global_centers.t());

    std::vector<int> mapping(local_k, -1);
    std::vector<char> used(global_k, 0);
    for (int c1 = 0; c1 < local_k; ++c1) {
        double best = std::numeric_limits<double>::infinity();
        int best_c2 = -1;
        for (int c2 = 0; c2 < global_k; ++c2) {
            if (used[c2])
                continue;
            double d = D(c1, c2);
            if (d < best) {
                best = d;
                best_c2 = c2;
            }
        }
        mapping[c1] = best_c2;
        if (best_c2 >= 0)
            used[best_c2] = 1;
    }

    for (int c1 = 0; c1 < local_k; ++c1) {
        if (mapping[c1] < 0 || mapping[c1] >= global_k)
            throw std::runtime_error(ctx + ": invalid local->global mapping");
    }

    return mapping;
}


// WCSS of a point subset using frozen original labels
static double fixed_label_wcss(
    const std::vector<int>& pts,
    const arma::mat& data,
    const arma::uvec& original_labels)
{
    if (pts.empty())
        return 0.0;

    std::vector<std::vector<int>> by_label(k);
    for (int p : pts)
        by_label[static_cast<int>(original_labels(p))].push_back(p);

    double wcss = 0.0;
    for (int c = 0; c < k; ++c)
    {
        const auto& grp = by_label[c];
        if (grp.empty()) continue;

        arma::rowvec centroid(d, arma::fill::zeros);
        for (int p : grp) centroid += data.row(p);
        centroid /= static_cast<double>(grp.size());

        for (int p : grp)
        {
            arma::rowvec diff = data.row(p) - centroid;
            wcss += arma::dot(diff, diff);
        }
    }
    return wcss;
}

// Construct a feasible component-to-anticluster assignment with the same
// randomized backtracking routine used by the VNS initializer.  Keeping this
// separate from the objective-based search lets the automatic T selector test
// candidates with precisely the same packing mechanism.
static bool try_construct_vns_start(
    const std::vector<std::vector<int>> &unit_cluster_count,
    const std::vector<int> &n_lb,
    const std::vector<int> &n_ub,
    int n_anticlusters,
    int attempts,
    std::mt19937 &rng,
    std::vector<int> *assignment_out = nullptr)
{
    const int n_units = static_cast<int>(unit_cluster_count.size());
    const int n_clusters = static_cast<int>(n_lb.size());
    if (n_anticlusters <= 0 || n_units < n_anticlusters ||
        n_clusters == 0 || static_cast<int>(n_ub.size()) != n_clusters)
        return false;

    for (const auto &counts : unit_cluster_count) {
        if (static_cast<int>(counts.size()) != n_clusters)
            return false;
    }

    std::vector<int> unit_total_size(n_units, 0);
    std::vector<int> unit_nonzero_clusters(n_units, 0);
    for (int u = 0; u < n_units; ++u) {
        for (int j = 0; j < n_clusters; ++j) {
            const int count = unit_cluster_count[u][j];
            if (count < 0)
                return false;
            unit_total_size[u] += count;
            if (count > 0)
                unit_nonzero_clusters[u]++;
        }
    }

    std::vector<int> ordered_units(n_units);
    std::iota(ordered_units.begin(), ordered_units.end(), 0);
    std::stable_sort(
        ordered_units.begin(), ordered_units.end(),
        [&](int lhs, int rhs) {
            if (unit_total_size[lhs] != unit_total_size[rhs])
                return unit_total_size[lhs] > unit_total_size[rhs];
            if (unit_nonzero_clusters[lhs] != unit_nonzero_clusters[rhs])
                return unit_nonzero_clusters[lhs] > unit_nonzero_clusters[rhs];
            return lhs < rhs;
        });

    struct BacktrackLevel {
        int unit = -1;
        std::vector<int> candidates;
        size_t next = 0;
        int chosen_anticluster = -1;
    };

    const int max_backtracks_per_attempt = std::max(1000, 20 * n_units);
    for (int attempt = 0; attempt < attempts; ++attempt) {
        std::vector<int> assignment(n_units, -1);
        std::vector<std::vector<int>> counts(
            n_clusters, std::vector<int>(n_anticlusters, 0));
        std::vector<BacktrackLevel> stack;
        stack.reserve(n_units);
        int depth = 0;
        int backtracks = 0;
        bool packed = false;

        while (true) {
            if (depth == n_units) {
                packed = true;
                break;
            }
            if (backtracks > max_backtracks_per_attempt)
                break;

            if (static_cast<int>(stack.size()) == depth) {
                const int unit = ordered_units[depth];
                BacktrackLevel level;
                level.unit = unit;
                level.candidates.reserve(n_anticlusters);
                for (int anticluster = 0; anticluster < n_anticlusters;
                     ++anticluster) {
                    bool fits = true;
                    for (int j = 0; j < n_clusters; ++j) {
                        if (counts[j][anticluster] + unit_cluster_count[unit][j] >
                            n_ub[j]) {
                            fits = false;
                            break;
                        }
                    }
                    if (fits)
                        level.candidates.push_back(anticluster);
                }
                std::shuffle(level.candidates.begin(), level.candidates.end(), rng);
                stack.push_back(std::move(level));
            }

            BacktrackLevel &level = stack[depth];
            if (level.next >= level.candidates.size()) {
                stack.pop_back();
                if (depth == 0)
                    break;
                depth--;
                backtracks++;
                BacktrackLevel &previous = stack[depth];
                if (previous.chosen_anticluster >= 0) {
                    const int unit = previous.unit;
                    const int anticluster = previous.chosen_anticluster;
                    assignment[unit] = -1;
                    for (int j = 0; j < n_clusters; ++j)
                        counts[j][anticluster] -= unit_cluster_count[unit][j];
                    previous.chosen_anticluster = -1;
                }
                continue;
            }

            const int anticluster = level.candidates[level.next++];
            assignment[level.unit] = anticluster;
            for (int j = 0; j < n_clusters; ++j)
                counts[j][anticluster] += unit_cluster_count[level.unit][j];
            level.chosen_anticluster = anticluster;
            depth++;
        }

        if (!packed)
            continue;

        bool feasible = true;
        for (int j = 0; j < n_clusters && feasible; ++j) {
            for (int anticluster = 0; anticluster < n_anticlusters;
                 ++anticluster) {
                if (counts[j][anticluster] < n_lb[j] ||
                    counts[j][anticluster] > n_ub[j]) {
                    feasible = false;
                    break;
                }
            }
        }
        if (!feasible)
            continue;

        if (assignment_out)
            *assignment_out = std::move(assignment);
        return true;
    }
    return false;
}

bool can_construct_vns_start(
    const std::vector<std::vector<int>> &unit_cluster_count,
    const std::vector<int> &n_lb,
    const std::vector<int> &n_ub,
    int n_anticlusters,
    int attempts,
    unsigned seed)
{
    std::mt19937 rng(seed);
    return try_construct_vns_start(
        unit_cluster_count, n_lb, n_ub, n_anticlusters, attempts, rng);
}

static std::vector<int> build_vns_assignment(
    const std::vector<std::vector<int>> &unit_cluster_count,
    const std::vector<std::vector<arma::rowvec>> &unit_centered_sum,
    const std::vector<int> &n_lb,
    const std::vector<int> &n_ub,
    int n_starts = 3,
    int max_trials = 20,
    int kmax = 5,
    int init_attempts = 10000,
    unsigned seed = 42,
    double seconds_per_start = 60.0,
    double *out_value = nullptr)
{

    struct VnsState {
        std::vector<int> a;
        std::vector<std::vector<int>> q;
        std::vector<std::vector<arma::rowvec>> s;
        double value = 0.0;
    };

    auto term = [](const arma::rowvec &v, int q) -> double {
        if (q == 0)
            return 0.0;
        return arma::dot(v, v) / static_cast<double>(q);
    };

    auto build_state = [&](const std::vector<int> &assign) -> VnsState {
        VnsState st;
        st.a = assign;
        st.q.assign(k, std::vector<int>(t, 0));
        st.s.assign(k, std::vector<arma::rowvec>(t, arma::rowvec(d, arma::fill::zeros)));
        for (int u = 0; u < m; ++u) {
            const int h = assign[u];
            for (int j = 0; j < k; ++j) {
                st.q[j][h] += unit_cluster_count[u][j];
                st.s[j][h] += unit_centered_sum[u][j];
            }
        }
        for (int j = 0; j < k; ++j)
            for (int h = 0; h < t; ++h)
                st.value += term(st.s[j][h], st.q[j][h]);
        return st;
    };

    auto feasible_counts = [&](const VnsState &st) -> bool {
        for (int j = 0; j < k; ++j) {
            for (int h = 0; h < t; ++h) {
                if (st.q[j][h] < n_lb[j] || st.q[j][h] > n_ub[j])
                    return false;
            }
        }
        return true;
    };

    auto move_delta = [&](const VnsState &st, int b, int c, int v) -> double {
        const int u = st.a[b];
        if (u == v || (c >= 0 && st.a[c] != v))
            return std::numeric_limits<double>::infinity();
        double delta = 0.0;
        for (int j = 0; j < k; ++j) {
            const int dn = unit_cluster_count[b][j] - (c >= 0 ? unit_cluster_count[c][j] : 0);
            const int qu = st.q[j][u] - dn;
            const int qv = st.q[j][v] + dn;
            if (qu < n_lb[j] || qu > n_ub[j] || qv < n_lb[j] || qv > n_ub[j])
                return std::numeric_limits<double>::infinity();
            arma::rowvec su = st.s[j][u];
            arma::rowvec sv = st.s[j][v];
            su -= unit_centered_sum[b][j];
            sv += unit_centered_sum[b][j];
            if (c >= 0) {
                su += unit_centered_sum[c][j];
                sv -= unit_centered_sum[c][j];
            }
            delta += term(su, qu) + term(sv, qv) - term(st.s[j][u], st.q[j][u]) - term(st.s[j][v], st.q[j][v]);
        }
        return delta;
    };

    auto apply_move = [&](VnsState &st, int b, int c, int v, double delta) {
        const int u = st.a[b];
        for (int j = 0; j < k; ++j) {
            const int dn = unit_cluster_count[b][j] - (c >= 0 ? unit_cluster_count[c][j] : 0);
            st.q[j][u] -= dn;
            st.q[j][v] += dn;
            st.s[j][u] -= unit_centered_sum[b][j];
            st.s[j][v] += unit_centered_sum[b][j];
            if (c >= 0) {
                st.s[j][u] += unit_centered_sum[c][j];
                st.s[j][v] -= unit_centered_sum[c][j];
            }
        }
        st.a[b] = v;
        if (c >= 0)
            st.a[c] = u;
        st.value += delta;
    };

    std::mt19937 rng(seed);
    auto random_int = [&](int n) { return std::uniform_int_distribution<int>(0, n - 1)(rng); };

    auto random_initial = [&](int attempts) -> VnsState {
        std::vector<int> assignment;
        if (!try_construct_vns_start(
                unit_cluster_count, n_lb, n_ub, t, attempts, rng, &assignment)) {
            throw std::runtime_error(
                "build_vns_assignment: no feasible random start found.");
        }
        return build_state(assignment);
    };

    auto local_search = [&](VnsState &st, const std::chrono::steady_clock::time_point &deadline) {
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        while (std::chrono::steady_clock::now() < deadline) {
            std::shuffle(order.begin(), order.end(), rng);
            bool improved = false;
            for (int b : order) {
                if (std::chrono::steady_clock::now() >= deadline)
                    break;
                for (int h = 0; h < t; ++h) {
                    const double delta = move_delta(st, b, -1, h);
                    if (delta < -1e-10) {
                        apply_move(st, b, -1, h, delta);
                        improved = true;
                        break;
                    }
                }
                if (improved)
                    break;
            }
            if (improved)
                continue;

            for (int i = 0; i < m && !improved; ++i) {
                if (std::chrono::steady_clock::now() >= deadline)
                    break;
                for (int h = i + 1; h < m; ++h) {
                    const int b = order[i];
                    const int c = order[h];
                    const int v = st.a[c];
                    const double delta = move_delta(st, b, c, v);
                    if (delta < -1e-10) {
                        apply_move(st, b, c, v, delta);
                        improved = true;
                        break;
                    }
                }
            }
            if (!improved)
                break;
        }
        st = build_state(st.a);
    };

    auto shake = [&](VnsState &st, int strength) {
        for (int step = 0; step < strength; ++step) {
            for (int trial = 0; trial < 100; ++trial) {
                const int b = random_int(m);
                const bool do_swap = (random_int(2) == 1);
                const int c = do_swap ? random_int(m) : -1;
                const int v = do_swap ? st.a[c] : random_int(t);
                const double delta = move_delta(st, b, c, v);
                if (std::isfinite(delta)) {
                    apply_move(st, b, c, v, delta);
                    break;
                }
            }
        }
    };

    VnsState best_global;
    bool found = false;
    int best_start_idx = -1;
    std::cout << "\nCreating anticluster partition with VNS (" << n_starts << " starts)\n";
    log_file << "\nCreating anticluster partition with VNS (" << n_starts << " starts)\n";
    for (int s = 0; s < n_starts; ++s) {
        VnsState best;
        try {
            best = random_initial(init_attempts);
        } catch (const std::exception &) {
            std::cout << "  Run " << (s + 1) << ": no feasible random start found, skipped\n";
            log_file << "  Run " << (s + 1) << ": no feasible random start found, skipped\n";
            continue;
        }
        const double start_value = best.value;
        const auto deadline = (seconds_per_start > 0.0)
                                  ? std::chrono::steady_clock::now() +
                                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                            std::chrono::duration<double>(seconds_per_start))
                                  : std::chrono::steady_clock::time_point::max();
        local_search(best, deadline);
        int k_cur = 1;
        for (int trial = 0; trial < max_trials && std::chrono::steady_clock::now() < deadline; ++trial) {
            VnsState y = best;
            shake(y, k_cur);
            local_search(y, deadline);
            if (y.value < best.value - 1e-10) {
                best = std::move(y);
                k_cur = 1;
            } else {
                k_cur = (k_cur == kmax) ? 1 : (k_cur + 1);
            }
        }
        if (!feasible_counts(best)) {
            std::cout << "  Run " << (s + 1) << ": infeasible after search, skipped\n";
            log_file << "  Run " << (s + 1) << ": infeasible after search, skipped\n";
            continue;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "  Run " << (s + 1) << ": " << start_value
                  << " -> " << best.value << "\n";
        log_file << std::fixed << std::setprecision(3)
                 << "  Run " << (s + 1) << ": " << start_value
                 << " -> " << best.value << "\n";
        if (!found || best.value < best_global.value) {
            best_global = std::move(best);
            found = true;
            best_start_idx = s;
        }
    }
    if (!found)
        throw std::runtime_error("build_vns_assignment: no feasible VNS solution found.");

    std::cout << "  Best anticluster partition: run " << (best_start_idx + 1)
              << " with D(P,S)=" << format_dps_value(best_global.value) << "\n";
    log_file << "  Best anticluster partition: run " << (best_start_idx + 1)
             << " with D(P,S)=" << format_dps_value(best_global.value) << "\n";
    if (out_value)
        *out_value = best_global.value;
    return best_global.a;
}




static arma::mat create_ac_partition_vns(
    const arma::mat& data,
    const arma::uvec& original_labels,
    const ConnectedComponents& cc,
    const arma::mat& global_mu,
    BResult& results)
{

    // -----------------------------------------------------------------
    // Assignment units:
    //   - one unit for each connected component induced by pairwise constraints;
    //   - one singleton unit for every point absent from that graph.
    //
    // Every unit must be assigned entirely to one anticluster.
    // -----------------------------------------------------------------

    std::vector<std::vector<int>> units;
    units.reserve(cc.comps.size() + n);

    std::vector<int> point_to_unit(n, -1);

    for (const auto& comp : cc.comps)
    {
        if (comp.empty())
            continue;

        const int u = static_cast<int>(units.size());
        units.push_back(comp);

        for (int p : comp)
        {
            if (p < 0 || p >= n)
            {
                throw std::runtime_error(
                    "create_ac_partition_vns: invalid point index "
                    + std::to_string(p) + " in connected component.");
            }

            if (point_to_unit[p] != -1)
            {
                throw std::runtime_error(
                    "create_ac_partition_vns: point "
                    + std::to_string(p)
                    + " appears in more than one connected component.");
            }
                point_to_unit[p] = u;
        }
    }

    for (int p = 0; p < n; ++p)
    {
        if (point_to_unit[p] != -1)
            continue;

        const int u = static_cast<int>(units.size());
        units.push_back({p});
        point_to_unit[p] = u;
    }

    const int n_units = static_cast<int>(units.size());

    if (n_units < t)
    {
        throw std::runtime_error(
            "create_ac_partition_vns: fewer assignment units than "
            "anticlusters; a nonempty partition is impossible.");
    }

    // -----------------------------------------------------------------
    // Cluster sizes.
    //
    // Requiring every anticluster to contain every original cluster
    // needs |C_j| >= T for every j.
    // -----------------------------------------------------------------

    std::vector<int> cluster_size(k, 0);

    for (int p = 0; p < n; ++p)
    {
        const int j = static_cast<int>(original_labels(p));

        if (j < 0 || j >= k)
        {
            throw std::runtime_error(
                "create_ac_partition_vns: invalid original label "
                + std::to_string(j) + " at point "
                + std::to_string(p) + ".");
        }

        cluster_size[j]++;
    }

    for (int j = 0; j < k; ++j)
    {
        if (cluster_size[j] < t)
        {
            throw std::runtime_error(
                "create_ac_partition_vns: cluster "
                + std::to_string(j)
                + " contains only "
                + std::to_string(cluster_size[j])
                + " points, fewer than T = "
                + std::to_string(t)
                + ". It cannot be represented in every anticluster.");
        }
    }

    // -----------------------------------------------------------------
    // Aggregate every assignment unit by original cluster.
    //
    // unit_cluster_count[u][j]:
    //     number of points of original cluster j in unit u.
    //
    // unit_centered_sum[u][j]:
    //     sum_{i in unit u intersect C_j} (p_i - mu_j).
    // -----------------------------------------------------------------

    std::vector<std::vector<int>> unit_cluster_count(
        n_units, std::vector<int>(k, 0));

    std::vector<std::vector<arma::rowvec>> unit_centered_sum;
    unit_centered_sum.reserve(n_units);

    for (int u = 0; u < n_units; ++u)
    {
        unit_centered_sum.emplace_back(
            k, arma::rowvec(d, arma::fill::zeros));

        for (int p : units[u])
        {
            const int j = static_cast<int>(original_labels(p));

            unit_cluster_count[u][j]++;

            unit_centered_sum[u][j] +=
                data.row(p) - global_mu.row(j);
        }
    }

    std::vector<int> n_lb(k, 0);
    std::vector<int> n_ub(k, 0);
    for (int j = 0; j < k; ++j)
    {
        const int lower = static_cast<int>(0.8 * (cluster_size[j] / t));
        const int upper = static_cast<int>(1.2 * (cluster_size[j] / t));
        n_lb[j] = lower;
        n_ub[j] = upper;
    }

    auto ws_start = std::chrono::steady_clock::now();
    double vns_value = -1.0;
    constexpr int vns_n_starts = 3;
    constexpr int vns_trials_fixed = 20;
    constexpr int vns_kmax_fixed = 5;
    constexpr int vns_init_attempts = 10000;
    constexpr unsigned vns_seed = 42;
    constexpr double vns_seconds_per_start = 60.0;
    std::vector<int> vns_assign = build_vns_assignment(
        unit_cluster_count, unit_centered_sum, n_lb, n_ub,
        vns_n_starts, vns_trials_fixed, vns_kmax_fixed,
        vns_init_attempts, vns_seed, vns_seconds_per_start, &vns_value);

    auto ws_end = std::chrono::steady_clock::now();

    arma::mat X_out(n, t, arma::fill::zeros);
    if (static_cast<int>(vns_assign.size()) != n_units) {
        throw std::runtime_error(
            "create_ac_partition_vns: invalid VNS assignment size.");
    }
    for (int u = 0; u < n_units; ++u) {
        const int h = vns_assign[u];
        if (h < 0 || h >= t) {
            throw std::runtime_error(
                "create_ac_partition_vns: invalid VNS anticluster id.");
        }
        for (int p : units[u])
            X_out(p, h) = 1.0;
    }

    results.vns_dps = vns_value;
    results.vns_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(ws_end - ws_start).count() / 1000.0;

    std::cout << "Using VNS anticluster partition.\n";
    log_file << "Using VNS anticluster partition.\n";
    std::cout << "  VNS_D(P,S): " << format_dps_value(results.vns_dps) << "\n";
    std::cout << "  VNS time: " << results.vns_time << " seconds\n";
    log_file << "  VNS_D(P,S): " << format_dps_value(results.vns_dps) << "\n";
    log_file << "  VNS time: " << results.vns_time << " seconds\n";

    return X_out;
}


// Use PC-SOS-SDP instance per anticluster to compute a global lower bound (LB / LB+)
std::vector<std::vector<std::vector<int> > > compute_pc_sos_sdp_lb(
    arma::mat &data,
    const UserConstraints &constraints,
    const std::vector<std::vector<int> > &ac_sol,
	const arma::uvec &original_labels,
    BResult &results) {

    auto *shared = new SharedDataPartition();
    shared->lb_ac.assign(t, 0.0);
    shared->lbplus_ac.assign(t, 0.0);
    shared->gap_ac.assign(t, 0.0);
    shared->sol_ac.resize(t);
    std::vector<int> cl_counts_per_subproblem = count_cl_per_subproblem(ac_sol, constraints);
    std::vector<int> points_per_subproblem;
    points_per_subproblem.reserve(t);
    shared->threadStates.reserve(n_threads_part);
    for (int i = 0; i < n_threads_part; ++i)
        shared->threadStates.push_back(false);

    for (int h = 0; h < t; ++h) {
        auto *job = new PartitionJob();

        // build data for anticluster using global indices stored in ac_sol[h]
        arma::uvec ac_idx(ac_sol[h].size());
        for (arma::uword ii = 0; ii < ac_idx.n_elem; ++ii)
            ac_idx(ii) = static_cast<arma::uword>(ac_sol[h][ii]);
        arma::mat data_ac = data.rows(ac_idx);
        UserConstraints ac_constraints = build_local_constraints(ac_idx, constraints);
        if (static_cast<int>(ac_constraints.cl_pairs.size()) != cl_counts_per_subproblem[h]) {
            throw std::runtime_error(
                "compute_pc_sos_sdp_lb: CL count mismatch between global recount and localized subproblem.");
        }
        points_per_subproblem.push_back(static_cast<int>(ac_sol[h].size()));

        // Get initial ub projecting the sol with fixed labels
        arma::mat fixed_sol(ac_sol[h].size(), k, arma::fill::zeros);
        for (int i = 0; i < ac_sol[h].size(); ++i)
        {
            int p = ac_sol[h][i];
            const int c = static_cast<int>(original_labels[p]);
            fixed_sol(i, c) = 1.0;
        }

        // Create job
        job->ac_id = h;
        job->ac_data = data_ac;
        job->ub_mssc = compute_mss(data_ac, fixed_sol);
        job->init_sol = fixed_sol;
        job->ac_constraints = ac_constraints;

        shared->queue.push_back(job);
        shared->print = true;
    }

    // solve with PC-SOS-SDP and store clustering results
    std::vector<std::vector<std::vector<int> > > sol_hc(
        t, std::vector<std::vector<int> >(k));
    ThreadPoolPartition p_pool(shared, n_threads_part);
    while (true) {
        {
            std::unique_lock<std::mutex> l(shared->queueMutex);
            while (!shared->worker_failed && is_thread_pool_working(shared->threadStates)) {
                shared->mainConditionVariable.wait(l);
            }
            if (shared->worker_failed)
                break;
            if (shared->queue.empty())
                break;
        }
    }

    p_pool.quitPool();
    if (shared->worker_failed) {
        std::string err = shared->worker_error;
        delete (shared);
        throw std::runtime_error("compute_pc_sos_sdp_lb: " + err);
    }

	// collect all the results
    for (auto &v: shared->lb_ac)
        results.lb += v;
    for (auto &v: shared->lbplus_ac)
        results.lb_plus += v;
    double gap_sum = 0.0;
    for (auto &v: shared->gap_ac)
        gap_sum += v;
    if (t > 0) {
        results.avg_gap = gap_sum / static_cast<double>(t);
        results.min_gap = *std::min_element(shared->gap_ac.begin(), shared->gap_ac.end());
        results.max_gap = *std::max_element(shared->gap_ac.begin(), shared->gap_ac.end());
        double cl_sum = 0.0;
        for (int v : cl_counts_per_subproblem)
            cl_sum += static_cast<double>(v);
        results.sum_cl_subproblem = cl_sum;
        results.avg_cl_subproblem = cl_sum / static_cast<double>(t);
        results.min_cl_subproblem = static_cast<double>(*std::min_element(cl_counts_per_subproblem.begin(), cl_counts_per_subproblem.end()));
        results.max_cl_subproblem = static_cast<double>(*std::max_element(cl_counts_per_subproblem.begin(), cl_counts_per_subproblem.end()));
    }
    {
        std::ostringstream oss_cl;
        for (size_t i = 0; i < cl_counts_per_subproblem.size(); ++i) {
            if (i > 0) oss_cl << ",";
            oss_cl << cl_counts_per_subproblem[i];
        }
        results.cl_per_subproblem = oss_cl.str();
    }
    {
        std::ostringstream oss_pts;
        for (size_t i = 0; i < points_per_subproblem.size(); ++i) {
            if (i > 0) oss_pts << ",";
            oss_pts << points_per_subproblem[i];
        }
        results.points_per_subproblem = oss_pts.str();
    }

    // save global idx of points per cluster
    // sol_ac[h] has shape (N_ac, F+1): first F cols = data, last col = cluster labels
    int hf = 0;
    for (auto &v: shared->sol_ac) {
        for (int c = 0; c < k; c++) {
            arma::uvec local_idx = arma::find(v.col(d) == c);
            sol_hc[hf][c].reserve(local_idx.n_elem);
            for (arma::uword li: local_idx)
                sol_hc[hf][c].push_back(ac_sol[hf][li]);
        }
        hf++;
    }

    // print files for each anticlusters label
    // for (int h = 0; h < t; ++h) {
    //     std::ofstream ac_file(result_path + "_ac_" + std::to_string(h) + ".txt");
    //     for (int c = 0; c < k; ++c) {
    //         for (const int p : sol_hc[h][c]) {
    //             ac_file << p << " " << c << "\n";
    //         }
    //     }
    //     ac_file.close();
    // }

    // free memory
    delete (shared);

    return sol_hc;
}


// Remount a final, feasible clustering solution for the whole data set
double mount_clustering_solution(
    arma::mat &data,
    const UserConstraints &constraints,
    const std::vector<std::vector<std::vector<int> > > &sol_hc,
    double heu_mss,
    arma::mat &heu_sol,
    arma::mat &final_sol,
	int &agreement) {

    // Build one global SDP instance: ML = points sharing the same local cluster
    // inside each anticluster; CL = all original cannot-link constraints.
    UserConstraints final_constraints = constraints;
    final_constraints.delta = 0.0;
    final_constraints.gamma = std::numeric_limits<double>::infinity();

    final_constraints.cl_pairs = constraints.cl_pairs;
    final_constraints.ml_pairs.clear();
    for (int h = 0; h < t; ++h) {
        for (int j = 0; j < k; ++j) {
            const std::vector<int> &pts = sol_hc[h][j];
            if (pts.empty()) continue;
            int i = pts[0];
            for (int b = 1; b < pts.size(); ++b) {
                int ii = pts[b];
                final_constraints.ml_pairs.emplace_back(i, ii);
            }
        }
    }

    sdp_branch_and_bound(k, data, heu_mss, final_constraints, final_sol, 1);

    if (final_sol.n_rows != static_cast<arma::uword>(n) || final_sol.n_cols != static_cast<arma::uword>(k)) {
        std::cout <<
            "mount_clustering_solution: sdp_branch_and_bound returned an unexpected "
            "solution size (" << std::to_string(final_sol.n_rows) << "x" <<
            std::to_string(final_sol.n_cols) << "), expected " << std::to_string(n) + "x" <<
            std::to_string(k) + ". The instance may be infeasible (check ml_pairs/cl_pairs "
            "for conflicts).";
        return -1.0;
    }

    // WCSS decomposition matching
    //   MSSC_B(O,K) = sum_t f(P*|S_t) + sum_j sum_t n_jt ||mu_jt - mu_j*||^2.
    arma::mat rm_centers = final_sol.t() * data;
    arma::vec counts = arma::sum(final_sol, 0).t();
    rm_centers.each_col() /= counts;

    std::vector<arma::uvec> idx_by_c(k);
    arma::mat mu(k, d, arma::fill::zeros);
    for (int j = 0; j < k; ++j) {
        idx_by_c[j] = arma::find(heu_sol.col(j) == 1.0);
        mu.row(j) = arma::sum(data.rows(idx_by_c[j]), 0) / idx_by_c[j].size();
    }
    std::vector<int> mapping = map_local_to_global_clusters(rm_centers, mu, "cloc final vs heu");

    arma::uvec heu_labels = arma::index_max(heu_sol, 1);
    arma::uvec rm_labels_local = arma::index_max(final_sol, 1);
    arma::mat rm_sol(n, k, arma::fill::zeros);
    arma::uvec rm_labels_mapped(n);
    for (int i = 0; i < n; ++i) {
        rm_labels_mapped(i) = mapping[rm_labels_local(i)];
        rm_sol(i, rm_labels_mapped(i)) = 1.0;
    }

    double rm_mss = compute_mss(data, rm_sol);

    int n_diff = static_cast<int>(arma::accu(heu_labels != rm_labels_mapped));
    agreement = static_cast<int>(arma::accu(heu_labels == rm_labels_mapped));

    if (n_diff == 0) {
        std::cout << "\n\nHeuristic and final solution are identical (up to cluster relabeling).\n";
        log_file << "\n\nHeuristic and final solution are identical (up to cluster relabeling).\n";
    } else {
        std::cout << "\n\nHeuristic and final solution differ in " << n_diff << " points.\n";
        log_file << "\n\nHeuristic and final solution differ in " << n_diff << " points.\n";
        return rm_mss;
    }

    arma::mat global_mu_final(k, d, arma::fill::zeros);
    for (int j = 0; j < k; ++j) {
        arma::uvec idx = arma::find(rm_sol.col(j) == 1.0);
        if (idx.n_elem == 0)
            continue;
        global_mu_final.row(j) = arma::sum(data.rows(idx), 0) / static_cast<double>(idx.n_elem);
    }

    double within = 0.0;
    double between = 0.0;

    for (int h = 0; h < t; ++h) {
        std::vector<std::vector<int>> pts_by_cluster(k);
        for (int j = 0; j < k; ++j) {
            for (int p : sol_hc[h][j])
                pts_by_cluster[rm_labels_mapped(p)].push_back(p);
        }

        for (int j = 0; j < k; ++j) {
            const std::vector<int>& pts = pts_by_cluster[j];
            const int n_jh = static_cast<int>(pts.size());
            if (n_jh == 0)
                continue;

            arma::uvec idx(n_jh);
            for (int r = 0; r < n_jh; ++r)
                idx(r) = static_cast<arma::uword>(pts[r]);

            arma::mat Xjh = data.rows(idx);
            arma::rowvec mu_jh = arma::mean(Xjh, 0);

            arma::mat diff = Xjh.each_row() - mu_jh;
            within += arma::accu(arma::square(diff));

            arma::rowvec diff_mu = mu_jh - global_mu_final.row(j);
            between += n_jh * arma::dot(diff_mu, diff_mu);
        }
    }

    double total_decomposed = within + between;

    std::cout << "\nsum_t f(P|S_t):  " << within;
    std::cout << "\nD(P,S): " << between;
    std::cout << "\nMSSC_B(O, K): " << total_decomposed;

    return rm_mss;
}


// run the anticlustering heuristic
void cap(arma::mat &data, UserConstraints &constraints, double heu_mss,
          arma::mat &heu_sol, ConnectedComponents &cc, BResult &results) {
	
    arma::uvec original_labels = arma::index_max(heu_sol, 1); // n-vector, global cluster id per point

    // Global centroids (from current heuristic labels), used as a robust fallback
    // when an anticluster has no samples for some global cluster.
    arma::mat global_mu(k, d, arma::fill::zeros);
    std::vector<int> global_count(k, 0);
    for (int p = 0; p < n; ++p) {
        int c = static_cast<int>(original_labels(p));
        if (c < 0 || c >= k) {
            throw std::runtime_error(
                "compute_pc_sos_sdp_lb: invalid label " + std::to_string(c) +
                " at point " + std::to_string(p) + " while building fallback centers.");
        }
        global_mu.row(c) += data.row(p);
        global_count[c]++;
    }
    for (int c = 0; c < k; ++c) {
        if (global_count[c] > 0)
            global_mu.row(c) /= global_count[c];
    }

	//-------------------- Create anticluster partition and lb+ ---------------------- //
    std::vector<std::vector<int> > ac_sol(t);
    std::cout << "\nCreating anticlustering partition\n";

    arma::mat current_ac_sol =
    create_ac_partition_vns(
        data,
        original_labels,
        cc,
        global_mu,
        results);

    // Convert the solved anticluster matrix into per-anticluster point lists.
    // This is the representation used downstream by the PC-SOS-SDP lower-bound routine.
    for (int h = 0; h < t; ++h) {
        arma::uvec idx = arma::find(current_ac_sol.col(h) == 1.0);
        ac_sol[h].reserve(idx.n_elem);
        for (arma::uword i = 0; i < idx.n_elem; ++i) {
            ac_sol[h].push_back(static_cast<int>(idx(i)));
        }
    }

    std::cout << "\n- Points per anticluster:\n";
    log_file << "\n- Points per anticluster:\n";
    for (int h = 0; h < t; ++h) {
        log_file << "  AC " << (h+1) << ": " << ac_sol[h].size() << " points\n";
        std::cout << "  AC " << (h+1) << ": " << ac_sol[h].size() << " points\n";
    }

    // ------------------ Compute lower bound using PC-SOS-SDP--------------------//
    auto start_time_lb = std::chrono::steady_clock::now();

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "Computing lower bound with PC-SOS-SDP\n\n";

    std::vector<std::vector<std::vector<int> > > sol_hc = compute_pc_sos_sdp_lb(data, constraints, ac_sol, original_labels, results);

    auto end_time_lb = std::chrono::steady_clock::now();
    results.lb_time =
            std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb)
            .count();

    // -------------------- Create feasible clustering solution -------------------- //
    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "Creating final remounted clustering solution\n\n";

    arma::mat final_sol(n, k, arma::fill::zeros);
    results.has_final_labels = false;
    results.final_labels.reset();
    try {
        int agreement_unused = 0;
        results.rm_mss = mount_clustering_solution(
            data, constraints, sol_hc, heu_mss, heu_sol, final_sol, agreement_unused);

        const bool valid_final_sol =
            (final_sol.n_rows == static_cast<arma::uword>(n)) &&
            (final_sol.n_cols == static_cast<arma::uword>(k)) &&
            arma::all(arma::sum(final_sol, 1) > 0.5);

        if (valid_final_sol) {
            results.final_labels = arma::index_max(final_sol, 1);
            results.has_final_labels = true;
        } else {
            results.rm_mss = 0.0;
        }
    } catch (...) {
        results.rm_mss = 0.0;
        results.has_final_labels = false;
        results.final_labels.reset();
    }

    // A normally completed run no longer needs the recovery checkpoint. If
    // the process is killed by the OOM killer, this point is never reached
    // and the last incumbent remains on disk.
    remove_incumbent_checkpoint();

}
