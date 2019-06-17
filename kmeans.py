import numpy as np

class KMeans:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.unknown = float('NaN')

    def cluster(self, features, attr_types, init_centroid_indexes=None):
        """
        Performs the clustering.

        Parameters:
        features (ndarray): The instances to be clustered
        attr_types (list): A list of the attribute type ('nominal' or 'continuous')
            corresponding to the columns in features
        init_centroid_indexes (list): You can pass in a list of indexes for the initial centroids
            (random if not provided)

        Returns:
        int: Number of clusters
        ndarray: The final centroids
        ndarray: A column of cluster assignments for the features
        ndarray: The final SSEs for each cluster
        float: The final total SSE

        """
        centroid_indexes = []
        if init_centroid_indexes is None:
            centroid_indexes = np.random.choice(np.arange(features.shape[0]), self.n_clusters)
        else:
            centroid_indexes = init_centroid_indexes

        centroids = features[centroid_indexes]

        last_sse = np.inf

        while True:
            cluster_sses = np.zeros((centroids.shape[0], ))
            assignments = np.zeros((features.shape[0], ))

            # Assign features to clusters
            for i,feature in enumerate(features):
                distances_to_centroids = []

                for centroid in centroids:
                    distance = self.get_distance(centroid, feature, attr_types)
                    distances_to_centroids.append(distance)

                closest_centroid_label = np.argmin(np.array(distances_to_centroids))
                assignments[i] = closest_centroid_label
                # We square the distance output to get to the SSE
                cluster_sses[closest_centroid_label] += distances_to_centroids[closest_centroid_label]**2


            # If the SSE hasn't changed, we are finished
            current_sse = np.sum(cluster_sses)
            if current_sse != last_sse:
                last_sse = current_sse
            else:
                break

            # Compute new centroids
            for i in range(centroids.shape[0]):
                features_in_cluster = features[np.where(assignments == i, True, False)]
                for j in range(centroids[i].shape[0]):
                    # If all unknown
                    if self.all_unknown(features_in_cluster, j):
                        centroids[i,j] = self.unknown
                    # If nominal
                    elif attr_types[j] == 'nominal':
                        centroids[i,j] = self.mode_wo_unknowns(features_in_cluster, j)
                    # If continuous
                    else:
                        centroids[i,j] = self.mean_wo_unknowns(features_in_cluster, j)

        silhouettes = self.silhouettes(centroids, assignments, features, attr_types)

        return centroids.shape[0], centroids, assignments, cluster_sses, last_sse, silhouettes

    def mode_wo_unknowns(self, features, col):
        column = features[:,col]
        col_wo_unknowns = column[np.where(np.isnan(column), False, True)]
        return np.argmax(np.bincount(col_wo_unknowns.astype(int)))

    def mean_wo_unknowns(self, features, col):
        column = features[:,col]
        col_wo_unknowns = column[np.where(np.isnan(column), False, True)]
        return np.mean(col_wo_unknowns)

    def all_unknown(self, features, col):
        return np.all(np.isnan(features[:,col]))

    def get_distance(self, centroid, feature, attr_types):
        unsquared = []
        for i in range(centroid.shape[0]):
            # One or both are unknown
            if np.isnan(centroid[i]) or np.isnan(feature[i]):
                unsquared.append(1)
            # Nominal
            elif attr_types[i] == 'nominal':
                unsquared.append(0 if feature[i] == centroid[i] else 1)
            # Continuous
            else:
                unsquared.append(feature[i] - centroid[i])

        return np.linalg.norm(np.array(unsquared))

    def silhouettes(self, centroids, assignments, features, attr_types):
        silhouettes = np.zeros_like(assignments, dtype=float)
        for i, feature in enumerate(features):
            other_features_in_cluster = features[np.where(assignments == assignments[i], True, False)]
            distances_wi_cluster = [self.get_distance(feature, other_feature, attr_types) for other_feature in other_features_in_cluster]
            a_i = sum(distances_wi_cluster)/len(distances_wi_cluster)

            closest_other_cluster_label = self.get_closest_cluster(feature, assignments[i], centroids, attr_types)
            other_features_in_cluster = features[np.where(assignments == closest_other_cluster_label, True, False)]
            distances_wi_cluster = [self.get_distance(feature, other_feature, attr_types) for other_feature in other_features_in_cluster]
            b_i = sum(distances_wi_cluster)/len(distances_wi_cluster)

            silhouettes[i] = (b_i - a_i)/max(a_i, b_i)

        return silhouettes


    def get_closest_cluster(self, feature, belongs_to, centroids, attr_types):
        distances = []
        for i, centroid in enumerate(centroids):
            if belongs_to == i:
                distances.append(np.inf)
            else:
                distances.append(self.get_distance(centroid, feature, attr_types))

        return np.argmin(np.array(distances))
