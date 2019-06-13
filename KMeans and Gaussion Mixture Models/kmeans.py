import numpy as np


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
             Finds n_cluster in the data x
             params:
                 x - N X D numpy array
             returns:
                 A tuple
                 (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
             Note: Number of iterations is the number of time you update the assignment
         '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        mu = np.random.choice(N, self.n_cluster)
        c = np.take(x, mu, axis=0)
        J = np.inf
        r = np.zeros(N)
        no = 0
        while no < self.max_iter:
            l2 = np.linalg.norm(x - np.expand_dims(c, axis=1), axis=2)
            r = np.argmin(l2, axis=0)
            J_new = np.sum([np.sum(np.linalg.norm(x[r==k] - c[k]) for k in range(self.n_cluster))]) / N
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            c = np.array([np.mean(x[r==k], axis=0) for k in range(self.n_cluster)])
            no=no+1
        return (c, r, no)
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting ((N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, updates = k_means.fit(x)
        votes = [{} for k in range(self.n_cluster)]
        for y1, r1 in zip(y, membership):
            if y1 not in votes[r1].keys():
                votes[r1][y1] = 1
            else:
                votes[r1][y1] += 1
        centroid_labels = []
        for votes_k in votes:
            if not votes_k:
                centroid_labels.append(0)
            centroid_labels.append(max(votes_k, key=votes_k.get))
        centroid_labels = np.array(centroid_labels)
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
        self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        l2 = np.sum(((x - np.expand_dims(self.centroids, axis=1)) ** 2), axis=2)
        r = np.argmin(l2, axis=0)
        labels=self.centroid_labels[r]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels