# Data and generated pairwise constraints

This directory contains the benchmark data and all files needed to reproduce the constrained CAP instances.

```text
data/
├── datasets/     		4 synthetic benchmark datasets in CAP input format
├── constraints/   		60 generated cannot-link (CL) files
├── labels/        		60 feasible PCCC reference label files
├── plots/         		plots generated for two-dimensional datasets
├── generator/          		the CL-instance generator
├── example_LOG.txt		log file for a synthetic instance
└── README.md			data formats, sources, and generation procedure
```

<br>

## 📄 Input formats

Every file in `datasets/` begins with

```text
N D K
```

where `N` is the number of observations, `D` the number of features, and `K`
the prescribed number of clusters. The remaining `N` lines are the feature
vectors.

Constraint files use zero-based observation indices:

```text
CL i j
```

The released instances contain cannot-link pairs. Each file in `labels/`
contains one zero-based cluster label per observation and is a feasible
reference clustering for the matching constraint set.

<br>

## 🧪 Benchmark datasets

The four synthetic datasets contain three equally sized Gaussian clusters in
two dimensions. Their names `syn-N-sigma` encode the number of observations
and the common within-cluster dispersion.

The real datasets are:

| Dataset | Configuration `(N,D,K)` | Source |
| --- | ---: | --- |
| Abalone | `(4177,10,3)` | [UCI](https://archive.ics.uci.edu/dataset/1/abalone) |
| EngyTime | `(4096,2,2)` | [Fundamental Clustering Problems Suite](https://data.mendeley.com/datasets/vsxvgc4rwy/1) |
| Mallat | `(2400,1024,4)` | [UCR Archive](https://www.timeseriesclassification.com/) |
| PR2392 | `(2392,2,3)` | [TSPLIB](https://www.math.uwaterloo.ca/tsp/history/tspinfo/pr2392_info.html) |
| Rice | `(3810,7,2)` | [UCI](https://archive.ics.uci.edu/dataset/545/rice) |
| Wifi | `(2000,7,5)` | [UCI](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) |

<br>

## 🛠️ Instance generation

`generator/create_feasible_instances.py` implements the generation procedure used
for the released instances.
It calls the [`PCCC algorithm`](https://github.com/phil85/PCCC-Algorithm),
a heuristic algorithm for clustering with hard and soft must-link and cannot-link constraints.
It creates constraint densities for both `p=20` and `p=40` across three seeds.

1. It runs K-means with 1,000 starts to obtain an unconstrained partition.
2. It exchanges points near the boundaries of close clusters, preserving each
   cluster cardinality, to form a nearby target partition.
3. It samples CL pairs whose endpoints have different target labels. The
   number of pairs is `floor(p * min_cluster_size / 100) * binom(K,2)`; sampling is biased toward
   points near a K-means decision boundary and reserves target-feasible pairs
   violated by the unconstrained partition.
4. It calls `pccc.py` to produce a feasible hard-CL reference clustering.
5. It writes the CL file, reference labels, K-means statistics, and, for
   two-dimensional data, a diagnostic plot.

From the repository root, run, for example:

```bash
python data/generator/create_feasible_instances.py \
       data/datasets/syn-1500-50.txt \
       data/constraints data/labels \
       --seed 1 --replica 1 --plots-dir data/plots --require-pccc
```

Install the dependencies first with
`python -m pip install -r data/generator/requirements.txt`. The `--require-pccc`
option enforces the procedure used for the released instances: generation
fails if PCCC does not return a feasible labeling.
