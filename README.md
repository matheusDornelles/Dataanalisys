# Data Analysis Course

A collection of data analysis assignments using Julia.

---

## Assignment 3 — K-Means Clustering on the Iris Dataset

This assignment applies unsupervised machine learning to the classic Iris dataset using the K-Means clustering algorithm.

### What it covers

**Section 1 – Exploratory Data Analysis (EDA)**  
Loads the Iris dataset (150 samples, 4 features, 3 species) and explores it through summary statistics, box plots per species, and a correlation heatmap to understand feature distributions and relationships.

**Section 2 – Pre-Processing & Standardisation**  
Builds the feature matrix and applies Z-score standardisation so that each feature has zero mean and unit variance. This is essential for K-Means since the algorithm is distance-based — features with larger raw variance would otherwise dominate cluster assignments.

**Section 3 – Choosing the Optimal Number of Clusters**  
Uses two complementary methods to find the best k:
- **Elbow method** — plots WCSS (within-cluster sum of squares) vs k and identifies the point of diminishing returns
- **Silhouette analysis** — measures how well each point fits its assigned cluster vs the nearest alternative cluster (computed for k = 2–10)

**Section 4 – Model Training & Cluster Profiling**  
Trains K-Means with the optimal k, reports convergence details, and profiles each cluster by computing centroids in both standardised and original feature space, plus a grouped bar chart of mean feature values per cluster.

### Run

```bash
julia assignment3/002362581_kmeans.jl
```

Output plots are saved to `assignment3/outputs/`.

### Dependencies

```
RDatasets, Clustering, Distances, StatsBase, StatsPlots, DataFrames
```
