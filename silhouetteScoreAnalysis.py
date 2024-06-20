from tslearn.clustering import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples
from tslearn.metrics import cdist_dtw, cdist_soft_dtw_normalized
from tslearn.utils import to_time_series_dataset, to_time_series
from scipy.spatial.distance import cdist


def silhouette_samples(X, labels, *, metric=None, **kwds):
    """Compute the Silhouette Coefficient for each sample.
    """
    sklearn_metric = None
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw" or metric is None:
        sklearn_X = cdist_dtw(X)
    elif metric == "softdtw":
        sklearn_X = cdist_soft_dtw_normalized(X)
    elif metric == "euclidean":
        X_ = to_time_series_dataset(X)
        X_ = X_.reshape((X.shape[0], -1))
        sklearn_X = cdist(X_, X_, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))

        def sklearn_metric(x, y):
            return metric(to_time_series(x.reshape((sz, d)),
                                         remove_nans=True),
                          to_time_series(y.reshape((sz, d)),
                                         remove_nans=True))
    metric = "precomputed" if sklearn_metric is None else sklearn_metric
    return sklearn_silhouette_samples(X=sklearn_X, 
                                      labels=labels, 
                                      metric=metric)

N = 50
filterSignal = np.array([2, 10, 11, 12, 18, 20, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 64, 66, 67, 69, 71, 72, 74, 75, 83, 87, 88, 93, 94, 95, 98, 106])
X = np.array([dataset(i) for i in range(1,N+1)])[:,270:,filterSignal+1]

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=2,
                          metric="dtw",
                          random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels, metric='dtw')
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is : {:.2f}"
        .format(silhouette_avg),
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()


        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Plot Silhouette untuk Beberapa Klaster")
    ax1.set_xlabel("Nilai Koefisien Silhouette")
    ax1.set_ylabel("Label Klaster")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Analisis Silhouette untuk Pengklasteran K-Means dengan n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(f'Analisis_Silhouette_Klaster_{n_clusters}.png')
plt.show()
