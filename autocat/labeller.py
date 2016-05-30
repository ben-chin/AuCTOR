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

        # training_data_idxs, _ = np.nonzero(p_matrix)
        # doc_vectors = self.docs_features[training_data_idxs]

        # Get classes/labels that apply to the documents
        # labels = self.labels_from_p_matrix(p_matrix[training_data_idxs])
        # label_vectors = MultiLabelBinarizer().fit_transform(labels)

        all_training_idxs = set()
        label_vectors = np.zeros(p_matrix.shape)
        for label in xrange(len(p_matrix[0])):
            doc_scores = p_matrix[:, label]
            n = 0.25 * len(doc_scores)
            training_idxs = self.get_n_best_examples(doc_scores, n)
            all_training_idxs.update(training_idxs)

            for i in training_idxs:
                label_vectors[i][label] = 1.0

        all_training_idxs = list(all_training_idxs)

        label_vectors = label_vectors[all_training_idxs]
        doc_vectors = self.docs_features[all_training_idxs]

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

    def get_n_best_examples(self, examples, n):
        return np.argsort(-examples)[:n]

    def labels_from_p_matrix(self, p_matrix):
        return map(lambda r: np.nonzero(r)[0], p_matrix)
