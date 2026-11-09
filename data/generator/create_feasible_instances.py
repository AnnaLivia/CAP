#!/usr/bin/env python3
"""Create feasible CL-only instances for CAP.

For each requested percentage p (by default 20 and 40), the script:
  1. computes a K-means partition z0 of the input data;
  2. creates a nearby feasible target partition z1 by swapping boundary
     points between close clusters;
  3. samples random cannot-link pairs whose endpoints have different labels
     in z1, including some pairs violated by z0;
  4. solves the CL instance with PCCC;
  5. saves the PCCC solution as the feasible solution certificate.  If PCCC
     is unavailable,
     the known target partition is used unless --require-pccc is specified.

The random pairs are selected only among pairs separated by z1, so the target
partition is feasible by construction.  The script also compares the final
feasible labels with the unconstrained K-means labels after the best global
relabeling.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


PERCENTAGES = (20,40)


def read_dataset(path: Path) -> tuple[np.ndarray, int, int, int]:
    """Read a dataset in the repository format: N D K followed by N rows."""
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()

    if len(header) != 3:
        raise ValueError(f"{path}: expected header 'N D K'.")

    n, d, k = map(int, header)
    data = np.loadtxt(path, dtype=float, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape != (n, d):
        raise ValueError(
            f"{path}: header declares ({n}, {d}), but data has shape {data.shape}."
        )
    if k < 2:
        raise ValueError("At least two clusters are required to create cannot-link pairs.")

    return data, n, d, k


def boundary_order(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return points of one cluster from closest to farthest from a boundary.

    The margin is the distance to the closest rival centroid minus the
    distance to the assigned centroid.  A small margin identifies a point
    close to a K-means decision boundary.  Random tie breaking is seeded.
    """
    squared_distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    own = squared_distances[np.arange(data.shape[0]), labels]
    squared_distances[np.arange(data.shape[0]), labels] = np.inf
    rival = np.min(squared_distances, axis=1)
    margin = rival - own

    members = np.flatnonzero(labels == cluster)
    tie_breaker = rng.random(members.size)
    return members[np.lexsort((tie_breaker, margin[members]))]


def partner_boundary_order(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    cluster: int,
    partner: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Order points in a cluster by proximity to a specific rival cluster."""
    members = np.flatnonzero(labels == cluster)
    own_distance = np.sum((data[members] - centers[cluster]) ** 2, axis=1)
    partner_distance = np.sum((data[members] - centers[partner]) ** 2, axis=1)
    margin = partner_distance - own_distance
    tie_breaker = rng.random(members.size)
    return members[np.lexsort((tie_breaker, margin))]


def closest_cluster_pairs(centers: np.ndarray) -> list[tuple[int, int]]:
    """Pair nearby clusters while ensuring that every cluster is included."""
    remaining = set(range(centers.shape[0]))
    pairs: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        best_pair: tuple[int, int] | None = None
        best_distance = np.inf
        for first in remaining:
            for second in remaining:
                if first >= second:
                    continue
                distance = float(np.sum((centers[first] - centers[second]) ** 2))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (first, second)
        if best_pair is None:
            break
        pairs.append(best_pair)
        remaining.remove(best_pair[0])
        remaining.remove(best_pair[1])

    # With an odd number of clusters, involve the unmatched cluster in a
    # second exchange with its closest already paired cluster.
    if remaining:
        unmatched = next(iter(remaining))
        partner = min(
            (cluster for cluster in range(centers.shape[0]) if cluster != unmatched),
            key=lambda cluster: float(
                np.sum((centers[unmatched] - centers[cluster]) ** 2)
            ),
        )
        pairs.append(tuple(sorted((unmatched, partner))))

    return pairs


def target_partition(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, set[int]]:
    """Create a feasible nearby partition by swapping boundary points.

    The swaps preserve every cluster cardinality.  Points are exchanged only
    between geometrically closest clusters and are selected near their common
    boundary, so the target partition remains a controlled perturbation of
    K-means while being different from it.
    """
    cluster_sizes = np.bincount(labels, minlength=centers.shape[0])
    cluster_pairs = closest_cluster_pairs(centers)
    if not cluster_pairs:
        raise ValueError("At least two nonempty clusters are required for swaps.")

    # Each exchange moves two points.  Allocate enough exchanges to move at
    # least 20% of the data while preserving every cluster cardinality.
    requested_moved = max(2, math.ceil(0.20 * labels.size))
    requested_swaps = max(len(cluster_pairs), math.ceil(requested_moved / 2))
    swaps_per_pair = [0] * len(cluster_pairs)
    points_used = np.zeros(centers.shape[0], dtype=int)

    # First assign one exchange to every pair, so every cluster is affected.
    for index, (first, second) in enumerate(cluster_pairs):
        if points_used[first] >= cluster_sizes[first] or points_used[second] >= cluster_sizes[second]:
            raise ValueError(
                "Cannot involve every cluster in a distinct swap while "
                "preserving cluster sizes."
            )
        swaps_per_pair[index] = 1
        points_used[first] += 1
        points_used[second] += 1

    # Distribute the remaining exchanges among pairs with unused points.
    for _ in range(requested_swaps - len(cluster_pairs)):
        available = [
            index
            for index, (first, second) in enumerate(cluster_pairs)
            if points_used[first] < cluster_sizes[first]
            and points_used[second] < cluster_sizes[second]
        ]
        if not available:
            max_moved = 2 * int(points_used.sum())
            raise ValueError(
                f"Cannot exchange at least 20% of the points while preserving "
                f"cluster sizes; the paired clusters allow at most {max_moved} points."
            )
        index = min(available, key=lambda candidate: swaps_per_pair[candidate])
        swaps_per_pair[index] += 1
        first, second = cluster_pairs[index]
        points_used[first] += 1
        points_used[second] += 1

    target = labels.copy()
    moved: set[int] = set()

    used_points: list[set[int]] = [set() for _ in range(centers.shape[0])]
    for (first, second), swap_count in zip(cluster_pairs, swaps_per_pair):
        if swap_count == 0:
            continue
        first_order = partner_boundary_order(
            data, labels, centers, first, second, rng
        )
        first_order = np.asarray(
            [point for point in first_order if point not in used_points[first]][:swap_count]
        )
        second_order = partner_boundary_order(
            data, labels, centers, second, first, rng
        )
        second_order = np.asarray(
            [point for point in second_order if point not in used_points[second]][:swap_count]
        )
        if len(first_order) < swap_count or len(second_order) < swap_count:
            raise ValueError("Not enough points to construct the target partition.")

        target[first_order] = second
        target[second_order] = first
        used_points[first].update(int(point) for point in first_order)
        used_points[second].update(int(point) for point in second_order)
        moved.update(int(point) for point in first_order)
        moved.update(int(point) for point in second_order)

    if np.array_equal(target, labels):
        raise ValueError("The target partition was not different from K-means.")
    if not np.array_equal(np.bincount(target, minlength=centers.shape[0]), cluster_sizes):
        raise AssertionError("Target partition changed cluster cardinalities.")
    return target, moved


def build_constraints(
    target_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    k: int,
    percentage: int,
    rng: np.random.Generator,
    boundary_probabilities: np.ndarray,
) -> list[tuple[int, int]]:
    """Sample random target-feasible cannot-link pairs.

    The percentage controls the number of pairs through the same density scale
    used by the instance generator.  The pairs themselves are sampled
    randomly by rejection from all unordered point pairs whose target labels
    are different.  The sampling is weighted toward points close to a
    K-means decision boundary.  A small fraction of the pairs is reserved for
    target-feasible pairs that violate the unconstrained K-means partition.
    """
    if kmeans_labels.shape != target_labels.shape:
        raise ValueError("K-means and target labels must have the same shape.")
    if boundary_probabilities.shape != target_labels.shape:
        raise ValueError("Boundary probabilities and labels must have the same shape.")
    if not np.isclose(boundary_probabilities.sum(), 1.0):
        raise ValueError("Boundary probabilities must sum to one.")

    cluster_sizes = np.bincount(target_labels, minlength=k)
    min_cluster_size = int(cluster_sizes.min())
    points_per_cluster_scale = (percentage * min_cluster_size) // 100
    n_constraints = max(1, points_per_cluster_scale * k * (k - 1) // 2)

    max_feasible_pairs = int(
        sum(
            cluster_sizes[first] * cluster_sizes[second]
            for first in range(k)
            for second in range(first + 1, k)
        )
    )
    if n_constraints > max_feasible_pairs:
        raise ValueError(
            f"p={percentage} requests {n_constraints} CL pairs, but only "
            f"{max_feasible_pairs} target-feasible pairs exist."
        )

    constraints: list[tuple[int, int]] = []
    selected: set[tuple[int, int]] = set()
    n = target_labels.size

    # Force a few target-feasible pairs to be violated by unconstrained
    # K-means. Such a pair has equal K-means labels and different target
    # labels, so no feasible solution can coincide with K-means.
    n_forced = max(1, n_constraints // 10)
    moved = np.flatnonzero(target_labels != kmeans_labels)
    moved_probabilities = boundary_probabilities[moved]
    if moved.size == 0 or moved_probabilities.sum() <= 0:
        raise ValueError(
            "The target partition does not provide a K-means-violating pair."
        )
    moved_probabilities = moved_probabilities / moved_probabilities.sum()

    attempts = 0
    max_attempts = max(1000, 100 * n_forced)
    while len(selected) < n_forced and attempts < max_attempts:
        attempts += 1
        first = int(rng.choice(moved, p=moved_probabilities))
        candidates = np.flatnonzero(
            (kmeans_labels == kmeans_labels[first])
            & (target_labels != target_labels[first])
        )
        candidates = candidates[candidates != first]
        if candidates.size == 0:
            continue
        candidate_probabilities = boundary_probabilities[candidates]
        if candidate_probabilities.sum() <= 0:
            continue
        candidate_probabilities = candidate_probabilities / candidate_probabilities.sum()
        second = int(rng.choice(candidates, p=candidate_probabilities))
        selected.add(tuple(sorted((first, second))))

    if len(selected) < n_forced:
        raise ValueError(
            "Could not construct enough K-means-violating but target-feasible "
            "cannot-link pairs."
        )

    while len(selected) < n_constraints:
        first = int(rng.choice(n, p=boundary_probabilities))
        second = int(rng.choice(n, p=boundary_probabilities))
        if first == second:
            continue
        if target_labels[first] == target_labels[second]:
            continue
        selected.add(tuple(sorted((first, second))))

    constraints.extend(sorted(selected))
    if len(constraints) != n_constraints:
        raise AssertionError("The generated random CL pairs have the wrong size.")
    if any(target_labels[first] == target_labels[second] for first, second in constraints):
        raise AssertionError("A generated CL pair is not feasible for target labels.")

    return constraints


def boundary_probabilities(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return sampling probabilities increasing near K-means boundaries."""
    squared_distances = np.sum(
        (data[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    two_nearest = np.partition(squared_distances, kth=1, axis=1)[:, :2]
    margins = two_nearest[:, 1] - two_nearest[:, 0]

    # A moderate exponential bias preserves randomness while favouring points
    # with a small distance gap between their two closest centroids.
    scale = float(np.percentile(margins, 75))
    scale = max(scale, np.finfo(float).eps)
    weights = np.exp(-2.0 * margins / scale)
    probabilities = weights / weights.sum()
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Could not compute finite boundary sampling probabilities.")
    return probabilities


def compare_with_kmeans(
    feasible_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    k: int,
) -> tuple[int, int, float, dict[int, int]]:
    """Compare two partitions after the best permutation of their labels."""
    if feasible_labels.shape != kmeans_labels.shape:
        raise ValueError("The two label vectors must have the same shape.")

    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (feasible_labels, kmeans_labels), 1)
    rows, columns = linear_sum_assignment(-confusion)
    relabeling = {int(column): int(row) for row, column in zip(rows, columns)}
    relabeled_kmeans = np.asarray(
        [relabeling[int(label)] for label in kmeans_labels], dtype=int
    )
    matches = int(np.count_nonzero(feasible_labels == relabeled_kmeans))
    mismatches = int(feasible_labels.size - matches)
    accuracy = matches / feasible_labels.size
    return matches, mismatches, accuracy, relabeling


def mssc_objective(data: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute the squared-Euclidean MSSC objective for a partition."""
    centers = np.empty((k, data.shape[1]), dtype=float)
    for cluster in range(k):
        members = data[labels == cluster]
        if members.shape[0] == 0:
            raise ValueError(f"Partition contains an empty cluster {cluster}.")
        centers[cluster] = members.mean(axis=0)
    return float(np.sum((data - centers[labels]) ** 2))


STATISTICS_FIELDS = (
    "dataset",
    "data_file",
    "percentage",
    "seed",
    "replica",
    "n",
    "d",
    "k",
    "n_constraints",
    "solution_source",
    "moved_boundary_points",
    "moved_boundary_percentage",
    "kmeans_cluster_sizes",
    "solution_cluster_sizes",
    "kmeans_objective_mssc",
    "pccc_objective_mssc",
    "solution_objective_mssc",
    "label_matches_after_relabeling",
    "label_mismatches_after_relabeling",
    "label_accuracy_after_relabeling",
    "cl_satisfied_by_kmeans",
    "cl_violated_by_kmeans",
    "cl_satisfaction_rate",
    "relabeling",
    "constraint_file",
    "label_file",
)


def write_statistics(path: Path, row: dict[str, object]) -> None:
    """Upsert one K-means comparison row in the statistics CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {field: existing.get(field, "") for field in STATISTICS_FIELDS}
                for existing in reader
            ]

    constraint_file = str(row["constraint_file"])
    rows = [
        existing
        for existing in rows
        if str(existing.get("constraint_file", "")) != constraint_file
    ]
    rows.append({field: row.get(field, "") for field in STATISTICS_FIELDS})

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATISTICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_constraints(path: Path, constraints: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for first, second in constraints:
            handle.write(f"CL {first} {second}\n")


def write_labels(path: Path, labels: np.ndarray) -> None:
    np.savetxt(path, labels.astype(int), fmt="%d")


def solve_with_pccc(
    data: np.ndarray,
    k: int,
    constraints: list[tuple[int, int]],
    seed: int,
) -> np.ndarray:
    """Solve the generated hard-CL instance with the repository's PCCC API."""
    from pccc import pccc

    result = pccc(
        data,
        k,
        ml=[],
        cl=constraints,
        sml=[],
        scl=[],
        sml_weights=[],
        scl_weights=[],
        metric="squared_euclidean",
        cluster_repositioning="violations_inertia",
        dynamic_n_neighbors="n_violations_neighbors.500.10.after_repositioning",
        random_state=seed,
    )
    if "labels" not in result:
        raise RuntimeError("PCCC did not return a labels vector.")
    return np.asarray(result["labels"], dtype=int)


def validate_solution(
    labels: np.ndarray,
    n: int,
    k: int,
    constraints: list[tuple[int, int]],
) -> None:
    """Validate a PCCC solution in the exact format expected by CAP."""
    if labels.shape != (n,):
        raise RuntimeError(f"PCCC returned labels with shape {labels.shape}, expected ({n},).")
    if np.any(labels < 0) or np.any(labels >= k):
        raise RuntimeError("PCCC returned a label outside the range [0, K).")
    if np.any(np.bincount(labels, minlength=k) == 0):
        raise RuntimeError("PCCC returned an empty cluster.")
    if any(labels[first] == labels[second] for first, second in constraints):
        raise RuntimeError("PCCC returned a solution violating a cannot-link constraint.")


def plot_coordinates(data: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Return two-dimensional coordinates used for the diagnostic plot."""
    if data.shape[1] != 2:
        raise ValueError("Plots are only defined for two-dimensional instances.")
    return data, "feature 1", "feature 2"


def write_plot(
    path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    constraints: list[tuple[int, int]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Save a scatter plot with cluster labels and cannot-link edges."""
    import matplotlib

    # The generator is intended to run on a VM without a graphical display.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))

    # Draw edges first so the labelled points remain visible.
    for first, second in constraints:
        axis.plot(
            coordinates[[first, second], 0],
            coordinates[[first, second], 1],
            color="black",
            linewidth=0.35,
            alpha=0.12,
            zorder=1,
        )

    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab20",
        s=16 if coordinates.shape[0] <= 3000 else 7,
        alpha=0.85,
        linewidths=0,
        zorder=2,
    )
    legend_handles, _ = points.legend_elements(prop="colors", num=None)
    axis.legend(
        legend_handles,
        [str(label) for label in sorted(np.unique(labels))],
        title="label",
        loc="best",
        fontsize="small",
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create feasible CL-only CAP instances for p=20 and p=40, "
            "then solve them with PCCC."
        )
    )
    parser.add_argument("data_file", type=Path, help="Dataset with header 'N D K'.")
    parser.add_argument("constraints_dir", type=Path, help="Directory for CL files.")
    parser.add_argument("labels_dir", type=Path, help="Directory for feasible label files.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic plots (default: sibling directory named plots).",
    )
    parser.add_argument(
        "--statistics-csv",
        type=Path,
        default=None,
        help=(
            "CSV file for K-means statistics (default: "
            "<constraints-dir>/../kmeans_cl_statistics.csv)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for K-means and tie breaking.")
    parser.add_argument(
        "--replica",
        type=int,
        default=None,
        help="Replica identifier used in filenames (default: the seed).",
    )
    parser.add_argument("--n-init", type=int, default=1000, help="Number of K-means starts.")
    parser.add_argument(
        "--require-pccc",
        action="store_true",
        help="Fail instead of using the known feasible target labels if PCCC fails.",
    )
    args = parser.parse_args()

    if args.n_init <= 0:
        parser.error("--n-init must be positive.")

    data, n, _d, k = read_dataset(args.data_file)
    statistics_csv = (
        args.statistics_csv
        or args.constraints_dir.parent / "kmeans_cl_statistics.csv"
    )
    model = KMeans(n_clusters=k, n_init=args.n_init, random_state=args.seed).fit(data)
    base_labels = model.labels_.astype(int)
    base_cluster_sizes = np.bincount(base_labels, minlength=k)
    if np.any(base_cluster_sizes == 0):
        raise RuntimeError("K-means returned an empty cluster.")
    kmeans_objective = mssc_objective(data, base_labels, k)
    boundary_sampling = boundary_probabilities(data, model.cluster_centers_)

    target_labels, moved = target_partition(
        data,
        base_labels,
        model.cluster_centers_,
        np.random.default_rng(args.seed),
    )
    target_cluster_sizes = np.bincount(target_labels, minlength=k)
    moved_percentage = 100.0 * len(moved) / n

    replica = args.seed if args.replica is None else args.replica
    args.constraints_dir.mkdir(parents=True, exist_ok=True)
    args.labels_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir or args.constraints_dir.parent / "plots"
    if _d == 2:
        plots_dir.mkdir(parents=True, exist_ok=True)
    stem = args.data_file.stem
    coordinates = x_label = y_label = None
    if _d == 2:
        coordinates, x_label, y_label = plot_coordinates(data)

    print(f"Dataset: {args.data_file} (N={n}, K={k})")
    print(f"K-means cluster sizes: {base_cluster_sizes.tolist()}")
    print(f"K-means MSSC objective: {kmeans_objective:.8f}")
    print(f"Target cluster sizes:   {target_cluster_sizes.tolist()}")
    print(f"Moved boundary points:  {len(moved)} ({moved_percentage:.2f}%)")

    for percentage in PERCENTAGES:
        rng = np.random.default_rng(args.seed + percentage)
        constraints = build_constraints(
            target_labels,
            base_labels,
            k,
            percentage,
            rng,
            boundary_sampling,
        )

        solution_source = "PCCC"
        try:
            solution_labels = solve_with_pccc(
                data, k, constraints, args.seed + percentage
            )
            validate_solution(solution_labels, n, k, constraints)
            solution_cluster_sizes = np.bincount(solution_labels, minlength=k)
        except Exception as error:
            if args.require_pccc:
                raise RuntimeError(
                    f"PCCC failed for p={percentage} and --require-pccc was set: {error}"
                ) from error
            solution_labels = target_labels.copy()
            solution_cluster_sizes = target_cluster_sizes.copy()
            solution_source = "target fallback"
            print(f"Warning: PCCC unavailable or invalid; using target labels ({error})")

        # Validate both the PCCC solution and the known feasible fallback before
        # writing the instance and comparing it with unconstrained K-means.
        validate_solution(solution_labels, n, k, constraints)
        solution_objective = mssc_objective(data, solution_labels, k)
        pccc_objective = solution_objective if solution_source == "PCCC" else None
        matches, mismatches, accuracy, relabeling = compare_with_kmeans(
            solution_labels,
            base_labels,
            k,
        )
        kmeans_cl_satisfied = sum(
            base_labels[first] != base_labels[second]
            for first, second in constraints
        )
        kmeans_cl_violated = len(constraints) - kmeans_cl_satisfied

        # Required convention: instance_number_percentage_replica.
        base = f"{stem}_{len(constraints)}_{percentage}_{replica}"
        constraint_path = args.constraints_dir / f"{base}_cl.txt"
        label_path = args.labels_dir / f"{base}.txt"
        plot_path = plots_dir / f"{base}.png"
        write_constraints(constraint_path, constraints)
        write_labels(label_path, solution_labels)
        write_statistics(
            statistics_csv,
            {
                "dataset": stem,
                "data_file": str(args.data_file),
                "percentage": percentage,
                "seed": args.seed,
                "replica": replica,
                "n": n,
                "d": _d,
                "k": k,
                "n_constraints": len(constraints),
                "solution_source": solution_source,
                "moved_boundary_points": len(moved),
                "moved_boundary_percentage": f"{moved_percentage:.8f}",
                "kmeans_cluster_sizes": ";".join(
                    str(size) for size in base_cluster_sizes
                ),
                "solution_cluster_sizes": ";".join(
                    str(size) for size in solution_cluster_sizes
                ),
                "kmeans_objective_mssc": f"{kmeans_objective:.8f}",
                "pccc_objective_mssc": (
                    f"{pccc_objective:.8f}" if pccc_objective is not None else ""
                ),
                "solution_objective_mssc": f"{solution_objective:.8f}",
                "label_matches_after_relabeling": matches,
                "label_mismatches_after_relabeling": mismatches,
                "label_accuracy_after_relabeling": f"{accuracy:.8f}",
                "cl_satisfied_by_kmeans": kmeans_cl_satisfied,
                "cl_violated_by_kmeans": kmeans_cl_violated,
                "cl_satisfaction_rate": (
                    f"{kmeans_cl_satisfied / len(constraints):.8f}"
                    if constraints
                    else "1.00000000"
                ),
                "relabeling": ";".join(
                    f"{predicted}->{reference}"
                    for predicted, reference in sorted(relabeling.items())
                ),
                "constraint_file": constraint_path.name,
                "label_file": label_path.name,
            },
        )
        if _d == 2 and coordinates is not None and x_label is not None and y_label is not None:
            write_plot(
                plot_path,
                coordinates,
                solution_labels,
                constraints,
                f"{stem} | {len(constraints)} CL | p={percentage} | replica={replica}",
                x_label,
                y_label,
            )

        print(
            f"p={percentage}: {len(constraints)} random CL pairs, "
            f"solution={solution_source}, sizes={solution_cluster_sizes.tolist()}\n"
            f"  MSSC objective: K-means={kmeans_objective:.8f}, "
            f"solution={solution_objective:.8f}\n"
            f"  comparison with unconstrained K-means after relabeling: "
            f"{matches} equal, {mismatches} different, accuracy={accuracy:.4f}\n"
            f"  relabeling K-means -> feasible: {relabeling}\n"
            f"  K-means on generated CL pairs: {kmeans_cl_satisfied} satisfied, "
            f"{kmeans_cl_violated} violated\n"
            f"  constraints: {constraint_path}\n"
            f"  labels:      {label_path}"
            f"\n  statistics:  {statistics_csv}"
            + (f"\n  plot:        {plot_path}" if _d == 2 else "")
        )


if __name__ == "__main__":
    main()
