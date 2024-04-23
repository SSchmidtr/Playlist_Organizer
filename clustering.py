import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load your data
songs = pd.read_csv('Total_Canciones.csv')
caracteristicas = pd.read_csv('Audio_Features.csv')

# Preprocess data
songs = songs.drop(columns=['Unnamed: 0', 'Playlist', 'Artist'])
caracteristicas = caracteristicas.drop(columns=['Unnamed: 0', 'valence', 
                                                'uri', 'analysis_url', 'time_signature', 'type', 'track_href'])
df = pd.merge(songs, caracteristicas, left_on='Track_ID', right_on='id')

# Prepare data for clustering
clustering_data = df.drop(columns=['Track', 'Track_ID', 'id'])

# Elbow method to find optimal number of clusters
wcss = []
cluster_range = range(1, 11)
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(clustering_data)
    wcss.append(kmeans.inertia_)

# Plot WCSS for elbow method
plt.plot(cluster_range, wcss, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Find silhouette scores for different cluster counts
silhouette_scores = []
for i in range(2, 11):  # Start from 2 clusters to avoid invalid silhouette scores
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(clustering_data)
    silhouette_score = metrics.silhouette_score(clustering_data, labels)
    silhouette_scores.append(silhouette_score)

# Plot silhouette scores to help determine optimal cluster count
plt.plot(range(2, 11), silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Clusters')
plt.show()


# Clustering algorithms to test
clustering_algorithms = {
    "K-Means": KMeans(n_clusters=9, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=9),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=9),
    "Spectral": SpectralClustering(n_clusters=9, random_state=42),
    "Gaussian Mixture": GaussianMixture(n_components=9, random_state=42),
}

# Test each clustering algorithm
results = []
for name, algorithm in clustering_algorithms.items():
    try:
        # Fit the clustering algorithm
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(clustering_data)
        else:
            algorithm.fit(clustering_data)
            labels = algorithm.labels_

        # Evaluate the clustering results
        silhouette_score = metrics.silhouette_score(clustering_data, labels)
        calinski_harabasz_score = metrics.calinski_harabasz_score(clustering_data, labels)
        davies_bouldin_score = metrics.davies_bouldin_score(clustering_data, labels)

        # Store the results
        results.append({
            'Algorithm': name,
            'Silhouette Score': silhouette_score,
            'Calinski-Harabasz Score': calinski_harabasz_score,
            'Davies-Bouldin Score': davies_bouldin_score,
        })
    except Exception as e:
        results.append({'Algorithm': name, 'Error': str(e)})

# Create a DataFrame to store the results
results_df = pd.DataFrame(results)

# Output results
print("Clustering Algorithm Performance:")
print(results_df)
