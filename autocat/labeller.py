import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer


class DocumentLabeller:

    def __init__(self, docs_features, rps_features):
        self.docs_features = docs_features
        self.rps_features = rps_features

    def label(self):
        sim = self.compute_similarity_matrix()
        p_matrix = self.compute_p_matrix(sim)

        training_data_idxs, _ = np.nonzero(p_matrix)
        doc_vectors = self.docs_features[training_data_idxs]

        # Get classes/labels that apply to the documents
        labels = self.labels_from_p_matrix(p_matrix[training_data_idxs])
        label_vectors = MultiLabelBinarizer().fit_transform(labels)

        return (doc_vectors, label_vectors)

    def compute_similarity_matrix(self):
        return linear_kernel(self.docs_features, self.rps_features)

    def compute_p_matrix(self, sim_matrix):
        # Compute confidence matrix
        for c in xrange(len(sim_matrix[0])):
            mx = max(sim_matrix[:, c])
            if mx > 0:
                sim_matrix[:, c] /= mx

        # Normalise over rows to find prob matrix
        for d in xrange(len(sim_matrix)):
            s = sum(sim_matrix[d, :])
            if s > 0:
                sim_matrix[d, :] /= s

        return sim_matrix

    def labels_from_p_matrix(self, p_matrix):
        return map(lambda r: np.nonzero(r)[0], p_matrix)
