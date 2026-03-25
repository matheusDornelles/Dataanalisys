using Random
using Statistics
using DataFrames
using StatsBase
using RDatasets
using Clustering
using Distances
using StatsPlots

const STUDENT_ID = "002362581"
const STUDENT_NAME = "Matheus Dornelles Barbosa Maia"

function safe_mean(v)
    isempty(v) ? NaN : mean(v)
end

function silhouette_per_point(distance_matrix::AbstractMatrix{<:Real}, labels::Vector{Int})
    n = length(labels)
    unique_labels = sort(unique(labels))
    s = zeros(Float64, n)

    for i in 1:n
        same = findall(labels .== labels[i])
        same_others = filter(j -> j != i, same)
        a = isempty(same_others) ? 0.0 : mean(distance_matrix[i, same_others])

        b = Inf
        for c in unique_labels
            c == labels[i] && continue
            other = findall(labels .== c)
            isempty(other) && continue
            b = min(b, mean(distance_matrix[i, other]))
        end

        if isinf(b) && a == 0.0
            s[i] = 0.0
        else
            denom = max(a, b)
            s[i] = denom == 0.0 ? 0.0 : (b - a) / denom
        end
    end

    return s
end

function elbow_k(k_values::Vector{Int}, y_values::Vector{Float64})
    x1, y1 = first(k_values), first(y_values)
    x2, y2 = last(k_values), last(y_values)

    distances = Float64[]
    for (x, y) in zip(k_values, y_values)
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = sqrt((y2 - y1)^2 + (x2 - x1)^2)
        push!(distances, num / den)
    end

    idx = argmax(distances)
    return k_values[idx]
end

function format_vector(v)
    join(round.(vec(v), digits=4), ", ")
end

println("Student ID: ", STUDENT_ID)
println("Student Name: ", STUDENT_NAME)
println("============================")
println("Section 1: Data Loading & EDA")
println("============================")

iris = dataset("datasets", "iris")
feature_syms = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
feature_names = string.(feature_syms)

println("Q1")
println("Dataset dimensions: ", size(iris))
println("Column names: ", names(iris))
class_counts = combine(groupby(iris, :Species), nrow => :Count)
println("Samples per species:")
show(class_counts, allrows=true, allcols=true)
println("\n")

println("Q2")
numeric_df = select(iris, feature_syms)
desc = DataFrame(
    Feature = feature_names,
    Mean = [mean(numeric_df[!, c]) for c in feature_syms],
    Std = [std(numeric_df[!, c]) for c in feature_syms],
    Min = [minimum(numeric_df[!, c]) for c in feature_syms],
    Max = [maximum(numeric_df[!, c]) for c in feature_syms]
)
show(desc, allrows=true, allcols=true)
println("\n")

mkpath("outputs")

default(size=(1100, 800))
p_box = plot(layout=(2, 2), legend=false)
for (i, f) in enumerate(feature_syms)
    boxplot!(p_box[i], string.(iris.Species), iris[!, f], xlabel="Species", ylabel=string(f), title=string(f))
end
savefig(p_box, "outputs/q3_boxplots_by_species.png")

function separation_score(values::Vector{<:Real}, groups)
    overall = mean(values)
    levels = unique(groups)
    between = 0.0
    total = sum((values .- overall).^2)
    for g in levels
        idx = findall(groups .== g)
        between += length(idx) * (mean(values[idx]) - overall)^2
    end
    total == 0.0 ? 0.0 : between / total
end

sep_scores = Dict{Symbol, Float64}()
for f in feature_syms
    sep_scores[f] = separation_score(iris[!, f], iris.Species)
end
best_feature = argmax(sep_scores)

println("Q3")
println("Saved box plots: outputs/q3_boxplots_by_species.png")
println("Feature with greatest inter-species separation (eta^2): ", best_feature)
println("Justification:")
println("The box plots show the clearest species separation on ", best_feature, ", with minimal overlap compared with the other features.")
println("Its between-species variance proportion is highest (eta^2 = ", round(sep_scores[best_feature], digits=4), "), indicating class means are far apart relative to within-class spread.")
println("This makes it the strongest single feature for distinguishing the three Iris species.")
println()

corr_mat = cor(Matrix(numeric_df))
p_heat = heatmap(feature_names, feature_names, corr_mat, c=:viridis, clim=(-1, 1), title="Q4 Correlation Heatmap", xlabel="Feature", ylabel="Feature")
savefig(p_heat, "outputs/q4_correlation_heatmap.png")

max_abs, max_pair = let max_abs = -Inf, max_pair = ("", "")
    for i in 1:length(feature_names)-1
        for j in i+1:length(feature_names)
            v = abs(corr_mat[i, j])
            if v > max_abs
                max_abs = v
                max_pair = (feature_names[i], feature_names[j])
            end
        end
    end
    (max_abs, max_pair)
end

println("Q4")
println("Saved heatmap: outputs/q4_correlation_heatmap.png")
println("Most strongly correlated pair: ", max_pair[1], " and ", max_pair[2], " (|r| = ", round(max_abs, digits=4), ")")
println("Implication: highly correlated features carry redundant information, so dimensionality reduction (e.g., PCA) can compress variance into fewer components with limited information loss.")
println()

println("===========================================")
println("Section 2: Pre-Processing & Standardisation")
println("===========================================")

X_df = select(iris, feature_syms)
has_missing = any(ismissing, eachcol(X_df))
X = permutedims(Matrix{Float64}(X_df))

println("Q5")
println("X size (features x samples): ", size(X))
println("Contains missing values: ", has_missing)
println()

means_before = mean(X, dims=2)
stds_before = std(X, dims=2)

z = fit(ZScoreTransform, X; dims=2)
Xz = StatsBase.transform(z, X)
means_after = mean(Xz, dims=2)
stds_after = std(Xz, dims=2)

println("Q6")
println("Before scaling means: [", format_vector(means_before), "]")
println("Before scaling stds : [", format_vector(stds_before), "]")
println("After scaling means : [", format_vector(means_after), "]")
println("After scaling stds  : [", format_vector(stds_after), "]")
println()

raw_long = DataFrame(Feature=String[], Value=Float64[])
std_long = DataFrame(Feature=String[], Value=Float64[])
for (idx, f) in enumerate(feature_names)
    append!(raw_long, DataFrame(Feature=fill(f, size(X, 2)), Value=vec(X[idx, :])))
    append!(std_long, DataFrame(Feature=fill(f, size(Xz, 2)), Value=vec(Xz[idx, :])))
end

p_v1 = @df raw_long violin(:Feature, :Value, legend=false, title="Raw Features", xlabel="Feature", ylabel="Value")
p_v2 = @df std_long violin(:Feature, :Value, legend=false, title="Standardised Features", xlabel="Feature", ylabel="Value")
p_violin = plot(p_v1, p_v2, layout=(1, 2), size=(1200, 450))
savefig(p_violin, "outputs/q7_violin_raw_vs_standardised.png")

println("Q7")
println("Saved violin plots: outputs/q7_violin_raw_vs_standardised.png")
println("Observation: raw features have different spread and scale, while standardised features are centered near 0 with comparable spread.")
println("Why it matters: K-Means uses distance, so unscaled high-variance features dominate cluster assignment; standardisation makes each feature contribute more equally.")
println()

std_pairs = collect(zip(feature_names, vec(stds_before)))
sorted_std = sort(std_pairs, by=x -> x[2], rev=true)
println("Q8")
println("Feature std deviations before scaling (descending): ", sorted_std)
println("Report note: omitting standardisation biases K-Means toward features with larger standard deviation; use the Q6 values as quantitative evidence.")
println()

println("====================================================")
println("Section 3: Choosing the Optimal Number of Clusters")
println("====================================================")

k_values = collect(1:10)
wcss = Float64[]
for k in k_values
    Random.seed!(42)
    km = kmeans(Xz, k; maxiter=500, tol=1e-8)
    push!(wcss, km.totalcost)
end

k_elbow = elbow_k(k_values, wcss)
p_elbow = plot(k_values, wcss, marker=:circle, xlabel="k", ylabel="WCSS", title="Q9 Elbow Method (WCSS vs k)", label="WCSS")
vline!(p_elbow, [k_elbow], linestyle=:dash, linewidth=2, label="Elbow ≈ k=$k_elbow")
annotate!(p_elbow, (k_elbow + 0.15, maximum(wcss) * 0.85, text("k=$k_elbow", 10)))
savefig(p_elbow, "outputs/q9_wcss_elbow.png")

println("Q9")
println("k -> WCSS:")
for (k, c) in zip(k_values, wcss)
    println("  ", k, " -> ", round(c, digits=4))
end
println("Saved elbow plot: outputs/q9_wcss_elbow.png")
println("Annotated elbow point: k = ", k_elbow)
println()

D = pairwise(Euclidean(), Xz; dims=2)
sil_k = Int[]
sil_avg = Float64[]
for k in 2:10
    Random.seed!(42)
    km = kmeans(Xz, k; maxiter=500, tol=1e-8)
    s = silhouette_per_point(D, km.assignments)
    push!(sil_k, k)
    push!(sil_avg, mean(s))
end

sil_df = DataFrame(k=sil_k, avg_silhouette=sil_avg)
sil_sorted = sort(sil_df, :avg_silhouette, rev=true)

p_sil = bar(sil_k, sil_avg, xlabel="k", ylabel="Average silhouette", title="Q10 Silhouette Scores", legend=false)
savefig(p_sil, "outputs/q10_silhouette_scores.png")

println("Q10")
println("Saved silhouette bar chart: outputs/q10_silhouette_scores.png")
println("Silhouette table sorted descending:")
show(sil_sorted, allrows=true, allcols=true)
println("\n")

best_sil_k = sil_sorted.k[1]
println("Q11")
println("Elbow optimises reduction in within-cluster variance (WCSS) as k increases, focusing on compactness and diminishing returns.")
println("Silhouette optimises both cohesion and separation, rewarding clusters that are internally tight and externally well-separated.")
println("For Iris, if the two disagree, silhouette is usually more trustworthy because it directly measures cluster quality relative to neighboring clusters, not only inertia reduction.")
println("Given the known three-species structure, a high silhouette near k=3 also aligns with domain expectations.")
println("Top silhouette k in this run: ", best_sil_k)
println()

println("Q12")
println("k=1 is excluded because silhouette needs at least one alternative cluster to compute the nearest-cluster distance b(i).")
println("With a single cluster, b(i) is undefined, so s(i) = (b(i)-a(i))/max(a(i),b(i)) cannot be evaluated in a meaningful finite way.")
println()

println("============================================")
println("Section 4: Model Training & Cluster Profiling")
println("============================================")

Random.seed!(42)
km3 = kmeans(Xz, 3; maxiter=500, tol=1e-8)

println("Q13")
if hasproperty(km3, :converged)
    println("Converged: ", getproperty(km3, :converged))
else
    println("Converged: not explicitly reported by this Clustering.jl version")
end
println("Iterations used: ", km3.iterations)
println("Total WCSS: ", round(km3.totalcost, digits=6))
println("Cluster sizes: ", km3.counts)
println()

centers_std = km3.centers
centers_orig = centers_std .* stds_before .+ means_before

println("Q14")
println("Centroids (standardised feature space):")
show(DataFrame(permutedims(centers_std), Symbol.(feature_names)), allrows=true, allcols=true)
println("\nCentroids (original feature space):")
show(DataFrame(permutedims(centers_orig), Symbol.(feature_names)), allrows=true, allcols=true)
println("\n")

cluster_labels = km3.assignments
original_with_cluster = DataFrame(select(iris, feature_syms))
original_with_cluster.Cluster = string.(cluster_labels)
cluster_means = combine(groupby(original_with_cluster, :Cluster), feature_syms .=> mean)

means_mat = Matrix(select(cluster_means, Not(:Cluster)))
renamed_cols = names(cluster_means)[2:end]
feature_labels_for_plot = replace.(renamed_cols, "_mean" => "")

p_bar = groupedbar(
    feature_labels_for_plot,
    means_mat',
    bar_position=:dodge,
    xlabel="Feature",
    ylabel="Mean (original scale)",
    title="Q15 Mean Feature Value by Cluster",
    label=permutedims("Cluster " .* cluster_means.Cluster)
)
savefig(p_bar, "outputs/q15_grouped_feature_means_by_cluster.png")

println("Q15")
println("Saved grouped bar chart: outputs/q15_grouped_feature_means_by_cluster.png")
println("Cluster-level means (original scale):")
show(cluster_means, allrows=true, allcols=true)
println("\nInterpretation: Cluster 1 is most clearly distinguished by petal measurements (PetalLength and PetalWidth), which tend to separate species groups more strongly than sepal features.")

println()
println("Completed. All requested outputs are in the outputs folder.")