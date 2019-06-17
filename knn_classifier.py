import numpy as np
import operator

class KNNClassifier:
    features = None
    labels = None

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, test_features, k=1, weight_distance=True, regression=False, heterogeneous=False, col_types=None):
        labels = []
        for test_feature in test_features:
            knn, knn_labels, knn_distances = self.get_knn(test_feature, k, heterogeneous=heterogeneous, col_types=col_types)
            pred = self.classify_by_neighbors(knn_labels, knn_distances, weight_distance=weight_distance) if not regression else self.regress_by_neighbors(knn_labels, knn_distances, weight_distance)
            labels.append(pred)

        return labels

    def regress_by_neighbors(self, labels, distances, weight_distance=True):
        summed_labels = np.sum(labels /
            ((distances**2 + .0000000001) if weight_distance else np.ones_like(labels)))
        divisor = np.sum(np.ones_like(distances) /
            ((distances**2 + .0000000001) if weight_distance else np.ones_like(distances)))
        return summed_labels / divisor

    def classify_by_neighbors(self, labels, distances, weight_distance=True):
        class_votes = dict()
        for i, distance in enumerate(distances):
            vote = 1/ ((distance**2 + .0000000001) if weight_distance else 1)
            if labels[i] in class_votes:
                class_votes[labels[i]] += vote
            else:
                class_votes[labels[i]] = vote

        # Return label
        return max(class_votes.items(), key=operator.itemgetter(1))[0]


    def get_knn(self, test_feature, k, heterogeneous=False, col_types=None):
        if not heterogeneous:
            distances = self.get_euclidean_distances(test_feature)
        elif col_types is None:
            print('No column types: using euclidean distance')
            distances = self.get_euclidean_distances(test_feature)
        else:
            distances = self.get_HEOMs(test_feature, col_types)

        min_indexes = distances.argsort()[:k]
        return self.features[min_indexes], self.labels.ravel()[min_indexes], distances[min_indexes]

    def get_euclidean_distances(self, test_feature):
        test_feature_col = np.tile(test_feature, (self.features.shape[0], 1))
        return np.linalg.norm(self.features - test_feature_col, axis=1)

    def get_HEOMs(self, test_feature, col_types):
        unsquared_distances = np.zeros_like(self.features)
        for i in range(test_feature.shape[0]):
            if col_types[i] == 'nominal':
                print(f'Features {i} is nominal')
                # Overlap Function
                unsquared_distances[:,i] = np.where(self.features[:,i] == test_feature[i], 0, 1)
            else:
                unsquared_distances[:,i] = self.features[:,i] - test_feature[i]

        distances = np.sum(np.abs(unsquared_distances)**2,axis=-1)**(1./2)

        return distances

    def is_nominal(self, col=0):
        nominal = self.unique_value_count(col) < self.features.shape[0]
        return nominal

    def unique_value_count(self, col=0):
        """
        Get the number of values associated with the specified attribute (or columnn)
        """
        values = len(np.unique(self.features[:,col]))
        return values
