from copyreg import dispatch_table
import numpy as np
from sklearn.neighbors import BallTree, NearestCentroid
from sklearn import svm
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform, cdist, pdist
from numpy.linalg import norm
from scipy import stats
import torch
import time
import matplotlib.pyplot as plt
from treelib import Tree

def lp_ball(radius, p, x1, centre):
    c1, c2 = centre
    if p == np.inf:
        x2_pos, x2_neg = radius * np.ones(len(x1)), -radius * np.ones(len(x1))
        return x2_pos, x2_neg
    pos_x2 = (radius**p - np.abs(x1 - c1)**p) ** (1 / p)
    neg_x2 = -pos_x2 + c2
    pos_x2 += c2
    return pos_x2, neg_x2


# class Patch(object):
#     def __init__(self, indices, radius, sample_size):
#         self.indices = indices
#         self.radius = radius
#         self.sample_size = sample_size
class Patch(object):
    def __init__(self, indices):
        self.indices = indices



#####################################################################################################################################
#                                                                                                                                   #
# Generalised ranked non-uniform ball tree                                                                                          #
#                                                                                                                                   #
#####################################################################################################################################
class ControlledRadiusRankedNUBT():
    """
        Non-uniform ball tree class that prioritises partition order by:
            1) Considering patches for which r_k / r_min >= c for an appropriately chosen c.
            2) Partitioning the patches with the largest point count.
        - The "radius control" aspect prevents balls in less denser regions from stagnating, being excluded from partitioning
        and therefore being captured with a poor resolution in the output solution. By controlling the ratio of the radii, we impose
        approximate uniformity across the patches.
        - Given the problematic patches that exceed the bound as in 1); we then prioritise by sample size N_k. In doing so, we
        target regions of high density in the distribution.

        By applying these two criteria, we follow the principle that we want our patches to follow the distribution of the big
        measure, while also exhibiting a roughly-uniform spatial partition.

        Note: 1) By filtering the patches through the radius control step, it is now possible a patch may contain 1 particle.
              From a statistical point of view this may be desirable. However, these clusters must be finalised at this point 
              to prevent the code from producing an error.
              2) A possible improvement may be that we assign the residual particles to a "complement set" that may then be eligible
              for clustering. To do so, some kind of radius handling is required (maybe we prioritise the non-clustered sets first?)

        INPUTS:
        big_measure :: the full particle measure to be partitioned via the ball tree
        n_clusters :: the number of patches we partition the full measure into; may contain different numbers of particles
        p :: the parameter p in the lp norm that is used to define a ball in the ball tree clustering.
    """
    def __init__(self, big_measure, n_clusters, radius_ratio_bound=1.5, p=2):
        self.big_measure = big_measure
        self.full_n_samples, self.n_features = big_measure.shape
        self.n_clusters = n_clusters
        self.p = p
        self.cluster_counter = 1
        self.radius_ratio_bound = radius_ratio_bound
        self.tree = Tree()
        self.tree.create_node(identifier="", data=Patch("Root"))
        self.tree.nodes[""].data.indices = np.arange(self.full_n_samples, dtype=int)
        self.stored_scores_dict = {"": self.full_n_samples}
        self.stored_rads_dict = {"": 1}
        self.branch_choices = ["0", "1"]


    def _radius_balance_hierarchy(self):
        """
            Perform the clustering on any subsets that violate the radius uniformity bound.
        """
        r_dict = {key: self.stored_rads_dict[key] for key, score in self.stored_scores_dict.items() if score > 1}
        r_min = np.copy(min(r_dict.values()))        
        violating_node_keys = [key for key, r_val in r_dict.items() if r_val / r_min >= self.radius_ratio_bound]

        while len(violating_node_keys) > 0:
            key = violating_node_keys[0]
            bm_inds = self.tree.nodes[key].data.indices
            if len(bm_inds) > 2: # Are different implementations faster for different input sizes?
                inds, rads, scores = self._ball_tree(key)
            else:
                inds, rads, scores = [0, 1], [0, 0], [1, 1]
            loc_keys = self._process_partition_data(key, inds, rads, scores, return_keys=True)
            if self.cluster_counter == self.n_clusters:
                return 1

            # Evaluate the new nodes
            for loc_key in loc_keys:
                if self.stored_scores_dict[loc_key] > 1 and self.stored_rads_dict[loc_key] / r_min >= self.radius_ratio_bound:
                    violating_node_keys.append(loc_key)
            violating_node_keys.pop(0)
        return 1
        

    def partition(self):
        """
            Partitions sets using the ball tree. The set chosen to be operated on is prioritised first to maintain uniformity with
            respect to some "radius" criteria, then on set size.
        """
        # ------------------------------------------------------------------------------------------------------------------------ #
        #                                                                                                                          #
        # Radius based choice                                                                                                      #
        #                                                                                                                          #
        # ------------------------------------------------------------------------------------------------------------------------ #
        if self.radius_ratio_bound > 1e-03:
            _ = self._radius_balance_hierarchy()
        if self.cluster_counter == self.n_clusters:
            return self.tree, self.stored_scores_dict.keys()


        # ------------------------------------------------------------------------------------------------------------------------ #
        #                                                                                                                          #
        # Score based choice                                                                                                       #
        #                                                                                                                          #
        # ------------------------------------------------------------------------------------------------------------------------ #
        # Split the cluster with the highest score, process the diagnostics of the output sets and test for completion
        branch = max(self.stored_scores_dict, key=self.stored_scores_dict.get)
        inds, rads, scores = self._ball_tree(branch)
        self._process_partition_data(branch, inds, rads, scores)
        if self.cluster_counter == self.n_clusters:
            return self.tree, self.stored_scores_dict.keys()
        return self.partition()


    def _ball_tree(self, branch, two_points=False):
        if two_points:
            return [np.array([0]), np.array([1])], [0, 0], [1, 1]

        max_inds = self.tree.nodes[branch].data.indices
        X = self.big_measure[max_inds]
        inds = np.arange(X.shape[0], dtype=int)
        centroid = np.mean(X, axis=0) # centroid = np.median(X, axis=0)
        X_dists = norm(centroid - X, ord=self.p, axis=1)
        X0_ind = np.argmax(X_dists)
        X0 = X[X0_ind]

        # bm_inds = self.tree.nodes[branch].data.indices
        # X_loc_dists = self.big_measure_dists[bm_inds][:, bm_inds]
        # X0_dists = X_loc_dists[X0_ind]
        # X1_ind = np.argmax(X0_dists)
        # X1 = X[X1_ind] # Closer to mean
        # X1_dists = X_loc_dists[X1_ind]
        # X_dists = self.big_measure_dists[bm_inds][:, bm_inds]
        # X0_dists = X_dists[X0_ind]

        X0_dists = norm(X - X0, axis=1, ord=self.p)
        X1_ind = np.argmax(X0_dists)
        X1 = X[X1_ind]
        X1_dists = norm(X - X1, axis=1, ord=self.p)

        mid_dist = np.median(X1_dists)
        # X1_mask = X1_dists <= mid_dist
        X1_mask = X1_dists < mid_dist
        inds_0, inds_1 = inds[~X1_mask], inds[X1_mask]
        r0, r1 = len(inds_0), len(inds_1)
        N0, N1 = len(inds_0), len(inds_1)
        # assert N0 > 0 and N1 > 0, f"N0 = {N0}, N1 = {N1} mid_dist = {mid_dist}, {X1_dists}"
        return [inds_0, inds_1], [r0, r1], [N0, N1]


    def _process_partition_data(self, branch, inds, rads, scores, return_keys=False):

        # Remove the data of the node we are going to operate on
        self.stored_scores_dict.pop(branch)
        self.stored_rads_dict.pop(branch)
        parent_id = self.tree.nodes[branch].identifier
        if self.cluster_counter == 1:
            parent_inds = np.arange(self.full_n_samples, dtype=int)
        else:
            parent_inds = self.tree.nodes[branch].data.indices
        
        # Organise the partition data
        lo_idx, hi_idx = 0, 1
        lo_inds, hi_inds = inds[lo_idx], inds[hi_idx]
        lo_rad, hi_rad = rads[lo_idx], rads[hi_idx]
        lo_score, hi_score = scores[lo_idx], scores[hi_idx]
        lo_branch, hi_branch = self.branch_choices[lo_idx], self.branch_choices[hi_idx]

        # Store this data under its associated key
        lo_inds = parent_inds[lo_inds]
        lo_branch = parent_id + lo_branch
        self.stored_scores_dict[lo_branch] = lo_score
        self.stored_rads_dict[lo_branch] = lo_rad
        self.tree.create_node(identifier=lo_branch, parent=parent_id, data=Patch(lo_inds))

        hi_inds = parent_inds[hi_inds]
        hi_branch = parent_id + hi_branch
        self.stored_scores_dict[hi_branch] = hi_score
        self.stored_rads_dict[hi_branch] = hi_rad        
        self.tree.create_node(identifier=hi_branch, parent=parent_id, data=Patch(hi_inds))

        # Add the new cluster to the count
        self.cluster_counter += 1
        if return_keys:
            return lo_branch, hi_branch


    def _direct_ball_tree(self, X, parent_node):
        """
            Along with the commented out section in _ball_tree, this may be quicker when dim is large.
            Note: comment out lo/hi_inds = parent_inds[lo/hi_inds] in _process_partition_data if using this.
        """
        centroid = np.mean(X, axis=0)
        X_dists = norm(centroid - X, ord=self.p, axis=1)
        X0_ind = np.argmax(X_dists)
        X0 = X[X0_ind]
        parent_inds = parent_node.data.indices

        X1_inds = self.BT.query(X0.reshape(1, -1), k=self.full_n_samples, return_distance=False)[0]
        mask = np.zeros(self.full_n_samples, dtype=int)
        mask[X1_inds] = 1
        mask[parent_inds] += 1
        X1_ind = np.arange(self.full_n_samples, dtype=int)[mask == 2][-1]
        X1 = self.big_measure[X1_ind]
        X1_dists = norm(X - X1, axis=1, ord=self.p)
        mid_dist = np.median(X1_dists)

        inds_1 = self.BT.query_radius(X1.reshape(1, -1), r=mid_dist)[0]
        mask = np.zeros(self.full_n_samples, dtype=int)
        mask[inds_1] = 1
        mask[parent_inds] += 1
        inds_1 = np.arange(self.full_n_samples, dtype=int)[mask == 2]
        mask = np.zeros(self.full_n_samples, dtype=int)
        mask[parent_inds] = 1
        mask[inds_1] += 1
        inds_0 = np.arange(self.full_n_samples, dtype=int)[mask == 1]

        r0, r1 = len(inds_0), len(inds_1)
        N0, N1 = len(inds_0), len(inds_1)
        return [inds_0, inds_1], [r0, r1], [N0, N1]


    def _compute_furthest_points(self, X):
        centroid = np.mean(X, axis=0)
        X1 = X[np.argmax(norm(centroid - X, ord=self.p, axis=1))]
        X2 = X[np.argmax(norm(X1 - X, ord=self.p, axis=1))]
        mid_dist = 0.5 * norm(X2 - X1, ord=self.p)
        return X1, X2, mid_dist


    def _plot_solution(self, clustered_measure=[]):
        self.colors = ["lightcoral", "chocolate", "darkorange", "gold"]
        self.colors = self.colors + ["yellowgreen", "limegreen", "turquoise", "dodgerblue"]
        self.colors = self.colors + ["cyan", "lightsteelblue", "magenta", "hotpink"]
        self.colors = self.colors + ["plum", "peru", "darkred", "burlywood"]
        if len(clustered_measure) > 0:
            fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
            axs[0].scatter(self.big_measure[:, 0], self.big_measure[:, 1], s=7, color="black")
            for n, val in enumerate(self.output_nodes):
                inds = self.tree.nodes[val].data.indices
                axs[0].scatter(self.big_measure[inds, 0], self.big_measure[inds, 1], s=7, color=self.colors[n])
                axs[1].scatter(self.big_measure[inds, 0], self.big_measure[inds, 1], s=7, color=self.colors[n])
                axs[1].scatter(clustered_measure[n, 0], clustered_measure[n, 1], s=7, color=self.colors[n], edgecolors="black")
            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")
            plt.show()
        else:
            fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
            ax.scatter(self.big_measure[:, 0], self.big_measure[:, 1], s=7, color="black")
            for n, val in enumerate(self.output_nodes):
                inds = self.tree.nodes[val].data.indices
                ax.scatter(self.big_measure[inds, 0], self.big_measure[inds, 1], s=7, color=self.colors[n])
            ax.set_aspect("equal")
            plt.show()


    def _ball_tree_svm(self, X, two_points=False):
        if two_points:
            return [np.array([0]), np.array([1])], [0, 0], [1, 1]
        n_samples = X.shape[0]


        # ------------------------------------------------------------------------------------------------------------------------ #
        #                                                                                                                          #
        # Point choice method                                                                                                      #
        #                                                                                                                          #
        # ------------------------------------------------------------------------------------------------------------------------ #
        # Convex hull method: chooses two furthest points (but isn't at the minute)
        # hull = ConvexHull(X)
        # hullpoints = X[hull.vertices, :]
        # hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        # bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        # X0_h, X1_h = X[bestpair[0]], X[bestpair[1]]
        
        # Direct computation method: chooses two furthest points
        # X_dists = squareform(pdist(X))
        # i_max, j_max = np.unravel_index(np.argmax(X_dists), X_dists.shape)
        # X0, X1 = X[i_max], X[j_max]

        # Mean based method:
        # centroid = np.mean(X, axis=0)
        centroid = np.median(X, axis=0)
        # centroid = np.array([np.median(X[:, 0]), np.median(X[:, 1])])
        X0 = X[np.argmax(norm(centroid - X, ord=self.p, axis=1))]
        X1 = X[np.argmax(norm(X0 - X, ord=self.p, axis=1))] # Closer to mean. This can be made quicker
        # mid_dist = 0.5 * norm(X1 - X0, ord=self.p)
        mid_dist = np.median(norm(X - X1, axis=1, ord=self.p))
        inds = np.arange(n_samples, dtype=int)
        X1_mask = norm(X - X1, axis=1) <= mid_dist
        inds_1 = inds[X1_mask]
        inds_0 = inds[~X1_mask]
        r0 = len(inds_0)
        r1 = len(inds_1)
        # r0 = np.max(norm(X[inds_0] - X_mean, axis=1, ord=self.p))
        # r1 = np.max(norm(X[inds_1] - X_mean, axis=1, ord=self.p))


        # ------------------------------------------------------------------------------------------------------------------------ #
        #                                                                                                                          #
        # Assignment method                                                                                                        #
        #                                                                                                                          #
        # ------------------------------------------------------------------------------------------------------------------------ #
        # Option A: Assign particles directly based on their distance to X_i
        # inds = np.arange(n_samples, dtype=int)
        # X0_dists = norm(X - X0, axis=1, ord=self.p)
        # X1_dists = norm(X - X1, axis=1, ord=self.p)
        # mask = X0_dists < X1_dists
        # inds_0 = inds[mask]
        # inds_1 = inds[~mask]
        # r0 = np.max(norm(X[inds_0] - X_mean, axis=1, ord=self.p))
        # r1 = np.max(norm(X[inds_1] - X_mean, axis=1, ord=self.p))

        # Option B: Assign particles using the ball tree, followed by direct assignment of the residual particles
        # mid_dist = 0.5 * norm(X1 - X0, ord=self.p)
        # inds_0, inds_1 = BallTree(X, metric="minkowski", p=self.p).query_radius(np.array([X0, X1]), r=mid_dist)
        # N0, N1 = len(inds_0), len(inds_1)
        # if N0 + N1 < n_samples:
        #     mask = np.ones(n_samples, dtype=bool)
        #     mask[inds_0], mask[inds_1] = False, False
        #     res_inds = np.arange(n_samples, dtype=int)[mask]
        #     X_res = X[res_inds]

        #     dist_mask = norm(X_res - X0, ord=self.p, axis=1) < norm(X_res - X1, ord=self.p, axis=1)
        #     X0_assigned_inds = res_inds[dist_mask]
        #     X1_assigned_inds = res_inds[~dist_mask]
        #     inds_0 = np.append(inds_0, X0_assigned_inds)
        #     inds_1 = np.append(inds_1, X1_assigned_inds)
        #     N0 += len(X0_assigned_inds)
        #     N1 += len(X1_assigned_inds)

        #     # Compute the generalised radii
        #     if len(X0_assigned_inds) > 0:
        #         r0 = np.max(norm(X0 - X[X0_assigned_inds], ord=self.p, axis=1))
        #     else:
        #         r0 = np.max(norm(X0 - X[inds_0], ord=self.p, axis=1))
        #     if len(X1_assigned_inds) > 0:
        #         r1 = np.max(norm(X1 - X[X1_assigned_inds], ord=self.p, axis=1))
        #     else:
        #         r1 = np.max(norm(X1 - X[inds_1], ord=self.p, axis=1))
        #     return [inds_0, inds_1], [r0, r1], [N0, N1]
        # r0 = np.max(norm(X0 - X[inds_0], ord=self.p, axis=1))
        # r1 = np.max(norm(X1 - X[inds_1], ord=self.p, axis=1))

        N0, N1 = len(inds_0), len(inds_1)
        return [inds_0, inds_1], [r0, r1], [N0, N1]



#####################################################################################################################################
#                                                                                                                                   #
# Ranked non-uniform ball tree                                                                                                      #
#                                                                                                                                   #
#####################################################################################################################################
class RankedNUBT():
    """
        Non-uniform ball tree class that partitions on clusters with the most particles first.
        full_n_samples :: the total number of particles to be partitioned; i.e. shape[0] of the input array.
        n_clusters :: the number of partitions to return. Should be a power of 2.
        n_features :: dimension of the space the particles are defined in.
        p :: the parameter p in the lp norm that is used to define a ball in the ball tree clustering.
    """
    def __init__(self, full_n_samples, n_clusters, n_features, p=2):
        self.full_n_samples = full_n_samples
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.p = p
        self.stored_scores_dict = np.zeros(n_clusters - 1)
        self.stored_clusters = []
        self.cluster_counter = 1
        self.residual_particles = np.zeros((full_n_samples, n_features))
        self.res_counter = 0
        self.NC = NearestCentroid()

    def measure_partition(self, X):
        """
            Driver code for performing the partition of the input particle measure.
            X :: array of the particles to be split with respect to the lp ball tree.
        """
        # Partition the input array and test for completion
        clusters, scores = self._ranked_non_uniform_partition(X)
        if self.cluster_counter == self.n_clusters:
            return clusters + self.stored_clusters

        # Otherwise, store the lower of the two scores to save computing this again, and store the corresponding sub-array
        lo_idx, hi_idx = np.argsort(scores)
        lo_score, hi_score = scores[lo_idx], scores[hi_idx]
        self.stored_scores_dict[self.cluster_counter - 2] = lo_score
        self.stored_clusters.append(clusters[lo_idx])
        old_max_idx = np.argmax(self.stored_scores_dict[:self.cluster_counter - 1])

        # In this case we can leave any old clusters as they are and directly partition the newly obtained largest sub-array
        if hi_score > self.stored_scores_dict[old_max_idx]:
            return self.measure_partition(clusters[hi_idx])
        
        # Otherwise, partition the stored cluster with the higher score and replace its data with the smaller new sub-array
        new_cluster = self.stored_clusters[old_max_idx]
        self.stored_scores_dict[old_max_idx] = hi_score
        self.stored_clusters[old_max_idx] = clusters[hi_idx]
        return self.measure_partition(new_cluster)


    def _ranked_non_uniform_partition(self, X, plot_lp_ball=False):
        """
            Partitions the particle set into two halves based on the centroid (i.e. the mean) 
            and two extreme points from this mean. The particles are initially classified by 
            an lp ball and the few remaining unclassified particles are assigned to the other half.

            The meaning of non uniform is that, in contrast to the partition method, 
            each new cluster is no longer guaranteed to contain the same number of particles
            as its counterpart, leading to clusters of different sizes throughout the global
            partitioning.

            X :: (N x n_features) array of particles, where 2 < N <= full_n_samples.
        """
        # Compute the furthest point from the centroid and then furthest from this point
        n_samples = X.shape[0]
        X1, X2, mid_dist = self._compute_furthest_points(X) ### Don't forget this possible cost gain
        self.cluster_counter += 1

        # If we get repeated samples of the same value then we simply approximately split into two
        if mid_dist == 0:
            threshold = n_samples // 2
            cluster_1 = X[:threshold]; cluster_2 = X[threshold:]
            N0, N1 = len(cluster_1), len(cluster_2)
            return [cluster_1, cluster_2], (N0, N1)

        double_classify = True
        # double_classify = False

        if double_classify:
            """
                In this method we cluster wrt to lp balls around 2 points and have residual particles. Tests have
                shown that assigning these using nearest centroid is more accurate than KDTree; so we try to make
                this step more efficient and accurate.
            """
            # Classify the data with respect to the two centroids based on a ball with radius the midpoint distance
            inds_0, inds_1 = BallTree(X, metric="minkowski", p=self.p).query_radius(np.array([X1, X2]), r=mid_dist)
            mask = np.ones(n_samples, dtype=bool)
            mask[inds_0], mask[inds_1] = False, False
            cluster_1, cluster_2 = X[inds_0], X[inds_1]
            N0, N1 = len(inds_0), len(inds_1)

            # We may have residual data that has not been assigned a cluster
            res_len = n_samples - (N0 + N1)
            X_res = X[np.arange(n_samples, dtype=int)[mask]]
            if res_len == 0 or len(X_res) == 0:
                return [cluster_1, cluster_2], (N0, N1)

            perform_NC = True
            # perform_NC = False
            # preserve_unclassified = True
            preserve_unclassified = False
            # centroid_dist = True
            centroid_dist = False

            if perform_NC:
                """
                    This is better in accuracy but performing on every cluster iteration seems excessive and possible to improve.
                """
                # Use the classified points as training data to classify the remaining points with nearest centroid. Can we possibly subsample here?
                # ind_long = np.zeros(N0 + N1, dtype=int)
                # X_long, ind_long[N0:] = np.concatenate((cluster_1, cluster_2), axis=0), 1
                # self.NC.fit(X_long, ind_long)
                # classes = self.NC.predict(X_res)
                # cluster_1 = np.append(cluster_1, X_res[classes == 0], axis=0)
                # cluster_2 = np.append(cluster_2, X_res[classes == 1], axis=0)
                # N0, N1 = len(cluster_1), len(cluster_2)
                smaller_cluster = np.argmin([N0, N1])
                if smaller_cluster == 0:
                    cluster_1 = np.append(cluster_1, X_res, axis=0)
                else:
                    cluster_2 = np.append(cluster_2, X_res, axis=0)
                N0, N1 = len(cluster_1), len(cluster_2)

            if preserve_unclassified:
                """
                    If True we store the residual particles for them to be assigned later only generationally using
                    NearestCentroid. Hopefully this saves on computational cost and improves accuracy.
                """
                if plot_lp_ball:
                    total_assigned = N0 + N1
                    self.lp_ball_test(X1, X2, mid_dist, cluster_1, cluster_2, total_assigned, n_samples, X)

                # Store the unclassified particles for later Nearest Centroid classification
                self.residual_particles[self.res_counter : self.res_counter + res_len] = np.copy(X_res)
                self.res_counter += res_len

                # Test if a generation is reached i.e. cluster counter is a power of 2
                if np.floor(np.log2(self.cluster_counter)) == np.ceil(np.log2(self.cluster_counter)):

                    # Consider the current clustered measure and the unassigned particles
                    clustered_measure = [cluster_1, cluster_2] + self.stored_clusters
                    residual_particles_loc = self.residual_particles[:self.res_counter]

                    # The number of training data is the same as the number of particles that have been assigned
                    n_assigned = self.full_n_samples - self.res_counter
                    classes, X_train = np.empty(n_assigned, dtype=int), np.empty((n_assigned, self.n_features))

                    # Assign the classes using the index identifying each cluster
                    m = 0
                    for n, X in enumerate(clustered_measure):
                        classes[m : m + len(X)] = n
                        X_train[m : m + len(X)] = X
                        m += len(X)
                    
                    # Record the entire data set for plotting/debugging
                    X_global = np.concatenate((X_train, residual_particles_loc), axis=0)

                    # Train NearestCentroid on the clustered data and predict the classes of the residual particles
                    self.NC = self.NC.fit(X_train, classes)
                    classes = self.NC.predict(residual_particles_loc)

                    # Extend each existing cluster with any related predict-assigned residual particles
                    X_assigned = np.empty((self.res_counter, self.n_features))
                    n_len = 0
                    for i in range(len(clustered_measure)):
                        ith_class_assigned = np.copy(residual_particles_loc[classes == i])
                        n_assigned = len(ith_class_assigned)
                        clustered_measure[i] = np.append(clustered_measure[i], ith_class_assigned, axis=0)
                        X_assigned[n_len : n_len + n_assigned] = ith_class_assigned
                        n_len += n_assigned
                    
                    # X_assigned_sort_inds = np.argsort(norm(X_assigned, axis=1))
                    X_assigned_sort = X_assigned[np.argsort(norm(X_assigned, axis=1))]
                    res_particles_sort = residual_particles_loc[np.argsort(norm(residual_particles_loc, axis=1))]
                    assert len(X_assigned_sort) == len(res_particles_sort), f"Length discrepancy: {len(X_assigned_sort)} {len(res_particles_sort)}"
                    for i in range(len(X_assigned_sort)):
                        assert X_assigned_sort[i, 0] == res_particles_sort[i, 0], f"Particle discrepancy"

                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
                    ax[0].scatter(X_global[:, 0], X_global[:, 1], color="black", s=5)
                    ax[0].scatter(X_train[:, 0], X_train[:, 1], color="aqua", s=5)
                    ax[0].set_title(f"Pre NC, Cluster Counter = {self.cluster_counter}")
                    ax[1].scatter(X_global[:, 0], X_global[:, 1], color="black", s=5)
                    ax[1].scatter(X_train[:, 0], X_train[:, 1], color="aqua", s=5)
                    ax[1].scatter(X_assigned[:, 0], X_assigned[:, 1], color="firebrick", s=5)
                    ax[1].set_title(f"Post NC, Cluster Counter = {self.cluster_counter}")
                    ax[0].set_aspect("equal")
                    ax[1].set_aspect("equal")
                    plt.show()

                    empirical_length = 0
                    for X in clustered_measure:
                        empirical_length += len(X)
                    assert empirical_length == self.full_n_samples, f"Particles seem to have gone missing: {empirical_length} {self.full_n_samples}"

                    # Reset the required variables
                    cluster_1, cluster_2, self.stored_clusters = clustered_measure[0], clustered_measure[1], clustered_measure[2:]
                    self.res_counter = 0
                    N0, N1 = len(cluster_1), len(cluster_2)
                    if plot_lp_ball:
                        self.lp_ball_test(X1, X2, mid_dist, cluster_1, cluster_2, N0 + N1, n_samples, X)


            if centroid_dist:
                """
                    This is a proposed alternative to NC by which residual points are classified based only on their distance from each centroid.
                """
                res_dists_1 = norm(X_res - X1, axis=1)
                res_dists_2 = norm(X_res - X2, axis=1)
                res_mask = res_dists_1 < res_dists_2
                cluster_1 = np.append(cluster_1, X_res[res_mask], axis=0)
                cluster_2 = np.append(cluster_2, X_res[np.invert(res_mask)], axis=0)
                N0, N1 = len(cluster_1), len(cluster_2)

        # var_1 = np.var(cluster_1, axis=0)
        # var_1 = np.dot(var_1, var_1)
        # var_2 = np.var(cluster_2, axis=0)
        # var_2 = np.dot(var_2, var_2)
        var_1 = 1
        var_2 = 1
        return [cluster_1, cluster_2], (var_1 * N0, var_2 * N1)


    def _compute_furthest_points(self, X):
        centroid = np.mean(X, axis=0)
        X1 = X[np.argmax(norm(centroid - X, ord=self.p, axis=1))]
        X2 = X[np.argmax(norm(X1 - X, ord=self.p, axis=1))]
        mid_dist = 0.5 * norm(X2 - X1, ord=self.p)
        return X1, X2, mid_dist


    def resample(self, clustered_measure):
        """
            Resample the clusters based on a resampling method. Currently this is by assigning
            a probability to each cluster that is proportional to the number of particles it
            contains.
        """
        # Define the probabilities proportional to the number of points in each cluster
        cluster_weights = np.array([len(X) for X in clustered_measure]) / self.full_n_samples

        # Sample the surviving clusters based on the empirical distribution
        sample = stats.rv_discrete(values=(range(self.n_clusters), cluster_weights)).rvs(size=self.n_clusters)
        X_new = np.array(clustered_measure, dtype=object)[sample]

        # Return the tensor obtained by iterating through each resampled cluster and randomly selecting a point
        return torch.Tensor(np.array([X[np.random.choice(len(X))] for X in X_new]))


    def lp_ball_test(self, X1, X2, mid_dist, cluster_1, cluster_2, total_assigned, n_samples, X):
            x1 = np.linspace(X1[0] - mid_dist, X1[0] + mid_dist, 2500)
            x2 = np.linspace(X2[0] - mid_dist, X2[0] + mid_dist, 2500)
            c1_pos, c1_neg = lp_ball(mid_dist, self.p, x1, X1)
            c2_pos, c2_neg = lp_ball(mid_dist, self.p, x2, X2)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(X[:, 0], X[:, 1], s=5, color="black")
            ax.scatter(cluster_1[:, 0], cluster_1[:, 1], color="palegreen", s=5)
            ax.scatter(cluster_2[:, 0], cluster_2[:, 1], color="aqua", s=5)
            ax.scatter(X1[0], X1[1], color="darkorange", s=7)
            ax.scatter(X2[0], X2[1], color="darkorange", s=7)
            ax.plot(x1, c1_pos, color="cornflowerblue", label=f"p={self.p}")
            ax.plot(x1, c1_neg, color="cornflowerblue")
            ax.plot(x2, c2_pos, color="cornflowerblue")
            ax.plot(x2, c2_neg, color="cornflowerblue")
            ax.set_aspect("equal")
            ax.set_title(f"{round(total_assigned / n_samples * 100, 1)}% assigned")
            ax.legend()
            ax.set_title(f"Cluster Counter = {self.cluster_counter}")
            # plt.savefig(f"BruteForce_ClusterCounter={self.cluster_counter}.png")
            # plt.savefig(f"NearestCentroid_ClusterCounter={self.cluster_counter}.png")
            plt.show()



#####################################################################################################################################
#                                                                                                                                   #
# Non-uniform ball tree                                                                                                             #
#                                                                                                                                   #
#####################################################################################################################################
class NUBT():
    def __init__(self, full_n_samples, n_clusters, n_features, p=2):
        self.full_n_samples = full_n_samples
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.completed_clusters = []
        self.outputs = []
        self.NC = NearestCentroid()
        self.n_completed = 0
        self.p = p

    def run_non_uniform_partition_2(self, X_list):
        """
            Alternative implementation of run_non_uniform_partition that seeks to 
            be more efficient by calling the expensive non_uniform_partition less frequently.

            X_list :: a list containing the current level disjoint clusters of the particle array.
        """
        n_incoming = len(X_list)
        n_missing = self.n_clusters - (self.n_completed + n_incoming)

        # This is the situation in which we can gain enough new clusters
        if n_missing - n_incoming <= 0:
            solns = [self._non_uniform_partition(X) for X in X_list[:n_missing]]
            [self.completed_clusters.extend(Y) for Y in solns]
            [self.completed_clusters.append(X) for X in X_list[n_missing:]]
            return self.completed_clusters

        # Otherwise keep partitioning the data
        X_list_new = []
        for X in X_list:
            soln = self._non_uniform_partition(X)
            for Y in soln:
                if len(Y) > 2:
                    X_list_new.append(Y)
                else:
                    self.completed_clusters.append(Y)
                    self.n_completed += 1
        return self.run_non_uniform_partition_2(X_list_new)


    def _non_uniform_partition(self, X, plot_lp_ball=False):
        """
            Partitions the particle set into two halves based on the centroid (i.e. the mean) 
            and two extreme points from this mean. The particles are initially classified by 
            a Euclidean ball and the remaining unclassified particles are assigned using NearestCentroid.
            Therefore this is currently brute force and could be improved to be made more efficient.

            The meaning of non uniform is that, in contrast to the partition method, 
            each new cluster is no longer guaranteed to contain the same number of particles
            as its counterpart, leading to clusters of different sizes throughout the global
            partitioning.

            X :: (N x n_features) array of particles, where 2 < N <= full_n_samples.
        """
        # Compute the furthest point from the centroid and then furthest from this point
        n_samples = X.shape[0]
        X1, X2, mid_dist = self._compute_furthest_points(X)

        # If we get repeated samples of the same value then this needs to be handled differently
        if mid_dist == 0:
            threshold = n_samples // 2
            cluster_1 = X[:threshold]; cluster_2 = X[threshold:]
            return cluster_1, cluster_2

        # Classify the data based on if it lies within a ball with radius the midpoint distance
        # inds_0, inds_1 = BallTree(X, metric="minkowski", p=self.p).query_radius(np.array([X1, X2]), r=mid_dist)
        inds_0 = BallTree(X, metric="minkowski", p=self.p).query_radius(X1.reshape((1, -1)), r=mid_dist)[0]
        mask = np.ones(n_samples, dtype=bool)
        mask[inds_0] = False
        inds_1 = np.arange(n_samples, dtype=int)[mask]
        cluster_1, cluster_2 = X[inds_0], X[inds_1]
        N0, N1 = len(inds_0), len(inds_1)
        total_assigned = N0 + N1
        # if plot_lp_ball:
            # return X1, X2, mid_dist, cluster_1, cluster_2, total_assigned
            # self.lp_ball_test(X1, X2, mid_dist, cluster_1, cluster_2, total_assigned, n_samples, X)
        # if total_assigned == n_samples:
            # return cluster_1, cluster_2

        # We have residual data that has not been assigned a cluster
        # mask = np.ones(n_samples, dtype=bool)
        # mask[inds_0], mask[inds_1] = False, False
        # X_res = X[np.arange(n_samples, dtype=int)[mask]]

        # # Use the classified points as training data to classify the remaining points with nearest centroid
        # ind_long = np.zeros(total_assigned, dtype=int)
        # X_long, ind_long[N0:] = np.concatenate((cluster_1, cluster_2), axis=0), 1
        # self.NC.fit(X_long, ind_long)
        # classes = self.NC.predict(X_res)
        # cluster_1 = np.append(cluster_1, X_res[classes == 0], axis=0)
        # cluster_2 = np.append(cluster_2, X_res[classes == 1], axis=0)
        return cluster_1, cluster_2


    def _compute_furthest_points(self, X):
        centroid = np.mean(X, axis=0)
        X1 = X[np.argmax(norm(centroid - X, ord=self.p, axis=1))]
        X2 = X[np.argmax(norm(X1 - X, ord=self.p, axis=1))]
        mid_dist = 0.5 * norm(X2 - X1, ord=self.p)
        return X1, X2, mid_dist


    def resample(self, clustered_measure):
        """
            Resample the clusters based on a resampling method. Currently this is by assigning
            a probability to each cluster that is proportional to the number of particles it
            contains.
        """
        # Define the probabilities proportional to the number of points in each cluster
        cluster_weights = np.array([len(X) for X in clustered_measure]) / self.full_n_samples

        # Sample the surviving clusters based on the empirical distribution
        sample = stats.rv_discrete(values=(range(self.n_clusters), cluster_weights)).rvs(size=self.n_clusters)
        X_new = np.array(clustered_measure, dtype=object)[sample]

        # Return the tensor obtained by iterating through each resampled cluster and randomly selecting a point
        return torch.Tensor(np.array([X[np.random.choice(len(X))] for X in X_new]))


    def deterministic_resample(self, clustered_measure):
        """
            Initial implementation of a deterministic resampling method, but problem still remains of how to
            fulfil the deficit without resorting to random sampling.
        """
        # Generate the weights for each cluster
        cluster_weights = np.array([len(X) for X in clustered_measure]) / self.full_n_samples

        # Compute the renewed count from each cluster by taking the integer part of the weight * n_clusters
        expectations = self.n_clusters * cluster_weights
        n_copies = np.floor(expectations).astype(int)
        deficit = self.n_clusters - np.sum(n_copies) # This is the missing number of resampled clusters

        # Compute the residuals from rounding the expectations and sample from the normalised distribution
        residuals = expectations - n_copies # These are the decimal parts that we've neglected
        residuals /= np.sum(residuals)
        deficit_compensation = stats.rv_discrete(values=(np.arange(self.n_clusters), residuals)).rvs(size=deficit)

        # Extract the samples
        cm = np.copy(np.array(clustered_measure, dtype=object))
        cm_new = []
        for m in range(self.n_clusters):
            for y in range(n_copies[m]):
                cm_new.append(cm[m])
        floor_sample = np.array([X[np.random.choice(len(X))] for X in cm_new])
        empirical_sample = np.array([X[np.random.choice(len(X))] for X in cm[deficit_compensation]])
        clustered_sample = np.concatenate((floor_sample, empirical_sample), axis=0)
        return torch.Tensor(clustered_sample)


    def run_non_uniform_partition(self, X_list):
        """
            Alternative implementation of run_non_uniform_partition that seeks to 
            be more efficient by calling the expensive non_uniform_partition less frequently.

            X_list :: a list containing the current level disjoint clusters of the particle array.
        """
        # Split the multiple clusters. With this approach, we only have multiple clusters being reprocessed
        X_list_new = []
        for X in X_list:
            # Assign each current level cluster in the list two new clusters
            soln = self._non_uniform_partition(X)
            for Y in soln:
                if len(Y) > 2:
                    X_list_new.append(Y)
                else:
                    self.completed_clusters.append(Y)
                    self.n_completed += 1

        # Ascertain the current missing number of clusters
        n_new_clusters = len(X_list_new)
        n_missing = self.n_clusters - (self.n_completed + n_new_clusters)
        if n_missing == 0:
            return self.completed_clusters + X_list_new
        
        # Test if the missing cluster figure is within reach
        local_missing = self.n_clusters - self.n_completed
        if local_missing > 2 * n_new_clusters:
            return self.run_non_uniform_partition(X_list_new)
        deficit = local_missing - n_new_clusters
        X_list_multiples = X_list_new[:deficit]
        self.completed_clusters.extend(X_list_new[deficit:])
        self.n_completed += n_new_clusters - deficit
        return self.run_non_uniform_partition(X_list_multiples)


    def run_non_uniform_partition_alt(self, X_list):
        """
            Alternative driver code for performing the non-balanced clustering.
            X_list: an iterable containing each of the particles we wish to partition
                    into two halves.
        """
        self.gen += 1
        X_list_new = []
        self.single_outputs.extend([X for X in X_list if len(X) <= 2])
        X_list_long = [X for X in X_list if len(X) > 2]
        for X in X_list_long:
            X_list_new.extend(self._non_uniform_partition(X))

        # Once we reach the critical generation we begin looking to fulfil the missing number of clusters
        if self.gen >= self.critical_gen:
            self.outputs.extend(X_list_new)

            # This parameter is really an expression of how many killed branches we've encountered
            deficit = self.n_clusters - (len(self.outputs) + len(self.single_outputs))
            if deficit > 1:
                # The most we can have is the number of clusters to double.
                threshold = deficit // 2

                # Extract the first half of the non-killed branches
                X = self.outputs[:threshold]
                self.outputs = self.outputs[threshold:]

                # Re-run the method on these clusters 
                self.run_non_uniform_partition_alt(X)
            elif deficit > 0:
                # If we are missing only one cluster, just perform the partition on one cluster from the eligible sets
                X = self.outputs[0]
                self.outputs = self.outputs[1:]
                self.run_non_uniform_partition_alt([X])
            return self.outputs + self.single_outputs
        return self.run_non_uniform_partition_alt(X_list_new)


    def lp_ball_test(self, X1, X2, mid_dist, cluster_1, cluster_2, total_assigned, n_samples, X):
            x1 = np.linspace(X1[0] - mid_dist, X1[0] + mid_dist, 2500)
            x2 = np.linspace(X2[0] - mid_dist, X2[0] + mid_dist, 2500)
            c1_pos, c1_neg = lp_ball(mid_dist, self.p, x1, X1)
            c2_pos, c2_neg = lp_ball(mid_dist, self.p, x2, X2)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(X[:, 0], X[:, 1], s=5, color="black")
            ax.scatter(cluster_1[:, 0], cluster_1[:, 1], color="palegreen", s=5)
            ax.scatter(cluster_2[:, 0], cluster_2[:, 1], color="aqua", s=5)
            ax.scatter(X1[0], X1[1], color="darkorange", s=7)
            ax.scatter(X2[0], X2[1], color="darkorange", s=7)
            ax.plot(x1, c1_pos, color="cornflowerblue", label=f"p={self.p}")
            ax.plot(x1, c1_neg, color="cornflowerblue")
            ax.plot(x2, c2_pos, color="cornflowerblue")
            ax.plot(x2, c2_neg, color="cornflowerblue")
            ax.set_aspect("equal")
            ax.set_title(f"{round(total_assigned / n_samples * 100, 1)}% assigned")
            ax.legend()
            plt.show()



#####################################################################################################################################
#                                                                                                                                   #
# Balanced ball tree                                                                                                                #
#                                                                                                                                   #
#####################################################################################################################################
class BalancedBallTree():
    def __init__(self, full_n_samples, n_clusters, n_features=2):
        """
            X_grad: tensor of shape (full_n_samples, n_features) that holds gradient values
            full_n_samples: the total number of particles in the measure
            n_clusters: the number of balls we partition into
            n_features: i.e. the number of dimensions
        """
        self.full_n_samples = full_n_samples
        self.n_clusters = n_clusters        
        if self.full_n_samples % n_clusters != 0:
            raise Exception("n_clusters must divide n_samples", self.full_n_samples % n_clusters)
        self.n_features = n_features
        self.n_points_pc = int(self.full_n_samples / n_clusters)
        self.top_count = 0
        self.X_out = np.zeros((n_clusters, self.n_points_pc, n_features))
        self.single_outputs = []
        self.outputs = []
        self.critical_gen = np.log2(n_clusters)
        assert 2 ** int(self.critical_gen) == n_clusters, "n_clusters should be a power of 2"
        self.gen = 0
        self.NC = NearestCentroid()


    def partition(self, X, sorting_inds=[]):
        """
            Recursively partitions the particle set X into n_clusters balls, 
            where the center of each ball is a randomly selected particle in 
            the subset the partitioning is being applied to, and the radius is 
            the median of the distances from all the other particles in the set 
            under consideration.
        """
        if len(sorting_inds) == 0:
            sorting_inds = np.arange(self.full_n_samples, dtype=int)

        # Test the local number of sample points
        if X.shape[0] > self.n_points_pc:
            inds, anti_inds = self._part_partition(X)
            X_near, X_far = X[inds], X[anti_inds]
            self.partition(X_near, sorting_inds[inds])
            self.partition(X_far, sorting_inds[anti_inds])
        else:
            self.X_out[self.top_count] = X
            # self.X_grads_out[self.top_count] = self.X_grads[sorting_inds]
            self.top_count += 1
        if self.top_count == self.n_clusters:
            # return self.X_out, self.X_grads_out
            return self.X_out


    def _part_partition(self, X):
        """
            Given a subset of X, randomly selects the point and partitions into two sets
            based on the median distance.
        """
        # Generate a random point from the sample
        n_samples = X.shape[0]
        X_sample = X[stats.randint(0, n_samples).rvs(size=1)]

        # Compute the median distance and the indices of any particles that share this distance
        med_dist, repeated_inds, dist_inds = self._compute_median_dist(X, X_sample)
        tree = BallTree(X)
        if len(repeated_inds) == 0:
            inds = tree.query_radius(X_sample, r=med_dist)[0]
            anti_inds = [ind for ind in range(n_samples) if ind not in inds]
            diff = len(inds) - len(anti_inds)
            if diff == 0:
                pass
            elif diff > 0:
                L = len(inds); M = len(anti_inds)
                anti_inds = np.append(anti_inds, inds[L - int(diff / 2): ])
                inds = inds[: L - int(diff / 2)]
            else:
                L = len(inds); M = len(anti_inds)
                abs_diff = np.abs(diff)
                inds = np.append(inds, anti_inds[M - int(abs_diff / 2): ])
                anti_inds = anti_inds[: M - int(abs_diff / 2)]
        else:
            half_n_samples = int(n_samples / 2)
            inds = np.zeros(half_n_samples, dtype=int)

            # Compute the sample indices corresponding to the points that lie inside the ball
            ind, dist = tree.query_radius(X_sample, r=med_dist, return_distance=True, sort_results=True)
            ind, dist = ind[0], dist[0]

            # We may have several instances of the median that aren't included in the ball.
            # Take the sorting index values to complete the index array
            ind_left = repeated_inds[repeated_inds < half_n_samples]
            ball_deficit = len(ind_left)

            # From the ball tree sorting index we know the indices to take that give particles with
            # values strictly less than the radius
            inds[:half_n_samples - ball_deficit] = ind[:half_n_samples - ball_deficit]

            # To complete the ball indices we extract the particle addresses stored in dist_inds
            inds[half_n_samples - ball_deficit:] = np.arange(X.shape[0], dtype=int)[dist_inds[ind_left]]
            anti_inds = [ind for ind in range(n_samples) if ind not in inds]

        assert len(inds) == len(anti_inds), f"index lengths are not equal: {len(inds)}, {len(anti_inds)}"
        assert (len(inds) % 2 == 0) and (len(anti_inds) % 2 == 0), f"index lengths are not mod 2: {len(inds)}, {len(anti_inds)}"
        return inds, anti_inds


    def _compute_median_dist(self, X, X_sample):
        """
            Given X_sample, computes the distances and the median of all points (including X_sample)
            from X_sample wrt the L2 norm. Sorts the distances and also returns the sorting index set.
            Since the median may be repeated, this is handles by returning indices of the sorting 
            index set that correspond to the particles with repeated median distance.
        """
        # Sort the distances and retain the sorting index
        dists = norm(X_sample - X, axis=1)
        dist_inds = np.argsort(dists)
        dists = dists[dist_inds]

        # Compute the median of the sorted distance values
        mr_ind = int(X.shape[0] / 2)
        ml_ind = mr_ind - 1
        mr = dists[mr_ind]
        ml = dists[ml_ind]
        med_dist = 0.5 * (ml + mr)

        # Track the indices of the sorting index array that possess the same median distance value
        repeated_inds = []
        if ml == mr:
            repeated_inds = np.arange(X.shape[0], dtype=int)[dists == med_dist]
        return med_dist, repeated_inds, dist_inds


    # def select_clusters(self, X, X_grads, distr):
    #     if distr == "unif":
    #         return torch.Tensor(X[:, 0]), torch.Tensor(X_grads[:, 0])



################################################
# TESTS TO GO STRAIGHT INTO CUBATURE ALGORITHM #
################################################
            # xlim = (np.min(clustered_measure[:, :, 0]), np.max(clustered_measure[:, :, 0]))
            # ylim = (np.min(clustered_measure[:, :, 1]), np.max(clustered_measure[:, :, 1]))
            # for n in range(n_clusters):
            #     fig, ax = plt.subplots()
            #     for k in range(n + 1):
            #         ax.scatter(clustered_measure[k, :, 0], clustered_measure[k, :, 1], s=5, color=colors[k])
            #     ax.scatter(clustered_measure[n + 1:, :, 0], clustered_measure[n + 1:, :, 1], s=5, color="black")
            #     plt.gca().set_aspect("equal")
            #     ax.set(xlim=xlim, ylim=ylim)
            #     plt.show()

        #    xlim = (np.min(clustered_measure[:, :, 0]), np.max(clustered_measure[:, :, 0]))
        #     ylim = (np.min(clustered_measure[:, :, 1]), np.max(clustered_measure[:, :, 1]))
        #     for n in range(n_clusters):
        #         fig, ax = plt.subplots(ncols=2)
        #         for k in range(n + 1):
        #             ax[0].scatter(clustered_measure[k, :, 0], clustered_measure[k, :, 1], s=5, color=colors[k])
        #             ax[1].scatter(clustered_grads[k, :, 0], clustered_grads[k, :, 1], s=5, color=colors[k])
        #         ax[0].scatter(clustered_measure[n + 1:, :, 0], clustered_measure[n + 1:, :, 1], s=5, color="black")
        #         ax[1].scatter(clustered_grads[n + 1:, :, 0], clustered_grads[n + 1:, :, 1], s=5, color="black")
        #         ax[0].set_aspect("equal")
        #         ax[1].set_aspect("equal")
        #         ax[0].set(xlim=xlim, ylim=ylim)
        #         # ax[1].set(xlim=xlim, ylim=ylim)
        #         plt.show()




            # Partition particles into subsets using ball tree
            # fig, axs = plt.subplots(ncols=2, figsize=(8, 8))
            # axs[0].scatter(cubature_measure[:, 0], cubature_measure[:, 1], s=5, color="black")
            # BTC = BallTreeCluster(cubature_grads, number_of_points, n_clusters, dimension)
            # clustered_measure = BTC.run_non_uniform_partition([np.array(cubature_measure)])
            # for cl in clustered_measure:
            #     axs[1].scatter(cl[:, 0], cl[:, 1], s=5)
            # axs[0].set_aspect("equal")
            # axs[1].set_aspect("equal")
            # plt.show()
            # cubature_measure = stats.multivariate_normal(mean=[i, 2 * i]).rvs(size=number_of_points)



###### Other stuff

            # assert np.abs(discrep) == 0, f"B branch: {inds}, {anti_inds}, {n_samples}, {mid_dist}"
            # assert np.abs(discrep) == 0, f"B branch: {X[inds]}, {X[anti_inds]}, {X[res_inds]}, {mid_dist}"
            # assert np.abs(discrep) == 0, f"B branch: {len(inds)}, {len(anti_inds)}, {n_samples}, {mid_dist}, {discrep}, {n}, {d1}, {d2}"
            # if np.abs(discrep) != 0:
            #     for i in range(len(X1)):
            #         if X1[i] != X2[i]:
            #             print(X1[i] - X2[i], X1[i], X2[i])
            # assert np.abs(discrep) == 0, f"Discrep = {discrep}, mid_dist = {mid_dist}"

       # for X in X_list:
        #     # Continue to partition the eligible subsets
        #     if X.shape[0] > 2:
        #         X_list_new.extend(self._non_uniform_partition(X))
        #     # If the subset can be no further clustered, store the results and kill the branch
        #     else:
        #         self.single_outputs.append(X)


                # for ind in inds:
            #     assert(norm(X[ind] - X1) <= mid_dist), "A"
            #     assert(norm(X[ind] - X2) > mid_dist), f"A1: {X1}, {X2}, {norm(X[ind])}, {norm(X[ind] - X2)}, {mid_dist}"
            # for anti_ind in anti_inds:
            #     assert(norm(X[anti_ind] - X2) <= mid_dist), f"B: {norm(X[anti_ind] - X2)}, {mid_dist}"
            #     assert(norm(X[anti_ind] - X1) > mid_dist), f"B1: {norm(X[anti_ind] - X2)}, {mid_dist}"
            # C = len(X[norm(X - X1, axis=1) <= mid_dist])
            # E = len(X[norm(X - X2, axis=1) <= mid_dist])
            # D = len(X[norm(X - X1, axis=1) > mid_dist]) - E
            # assert C + D + E == n_samples, f"Summation issue: {C + D + E}, {n_samples}"
            # anti_ball1 = norm(X[anti_inds] - X1, axis=1) == mid_dist # These are all supposed to be false
            # anti_ball2 = norm(X[inds] - X2, axis=1) == mid_dist # These are all supposed to be false
            # n = 0
            # for x in anti_ball1:
            #     if x:
            #         print("Ball 1:", norm(X[anti_inds[n]] - X1), mid_dist, x, norm(X[anti_inds[n]] - X1) == mid_dist)
            #     n += 1
            # m = 0
            # for x in anti_ball2:
            #     if x:
            #         print("Ball 2:", norm(X[inds[m]] - X2), mid_dist, x)
            #     n += 1
            #     m += 1


# assert 2 * n_doubles + len(self.single_outputs) == self.n_clusters, "Lengths"

        # if single_classify:
        #     """
        #         Note that with this method we have no residual particles; we just naively assign them all to the
        #         "other" cluster.
        #     """
        #     # Classify with respect to one of the centroids and default assign all the residual data to the other cluster
        #     inds_0 = BallTree(X, metric="minkowski", p=self.p).query_radius(X1.reshape((1, -1)), r=mid_dist)[0]
        #     mask = np.ones(n_samples, dtype=bool)
        #     mask[inds_0] = False
        #     inds_1 = np.arange(n_samples, dtype=int)[mask]
        #     cluster_1, cluster_2 = X[inds_0], X[inds_1]
        #     N0, N1 = len(inds_0), len(inds_1)


    # def measure_partition_double_rank(self, X):
    #     """
    #         Driver code for performing the partition of the input particle measure based on both cluster size and variance.
    #         X :: array of the particles to be split with respect to the lp ball tree.
    #     """
    #     # Partition the input array and test for completion
    #     clusters, scores = self._ranked_non_uniform_partition(X)
    #     if self.cluster_counter == self.n_clusters:
    #         return clusters + self.stored_clusters

    #     # Otherwise, store the lower of the two scores to save computing this again, and store the corresponding sub-array
    #     lo_idx, hi_idx = np.argsort(scores)
    #     lo_score, hi_score = scores[lo_idx], scores[hi_idx]
    #     self.stored_scores_dict[self.cluster_counter - 2] = lo_score
    #     self.stored_clusters.append(clusters[lo_idx])

    #     # We're always going to need to take the highest value from the stored list...
    #     old_max_idx = np.argmax(self.stored_scores_dict)
    #     stored_max = self.stored_scores_dict[old_max_idx]
    #     self.stored_scores_dict[old_max_idx] = 0 # Temporary hack

    #     # ... so we only need to compare our recent hi_score with the rest of the list
    #     old_second_max_idx = np.argmax(self.stored_scores_dict)
    #     stored_second_max = self.stored_scores_dict[old_second_max_idx]
    #     self.stored_scores_dict[old_max_idx] = stored_max

    #     # Test hi_score against the 2nd max in the storeds to complete the duo
    #     max_var = np.var(self.stored_clusters[old_max_idx], axis=0)
    #     if hi_score > stored_second_max:
    #         """
    #             In this case the duo is the hi_score cluster and the old_max_idx cluster
    #         """            
    #         # Do the variance tests between the hi_score cluster and the old_max_idx cluster
    #         hi_var = np.var(clusters[hi_idx], axis=0)
    #         if np.dot(hi_var, hi_var) > np.dot(max_var, max_var):
    #             return self.measure_partition_double_rank(clusters[hi_idx])
            
    #         # If we act on the stored cluster we need to replace its data with that of the hi_score cluster
    #         X0 = np.copy(self.stored_clusters[old_max_idx])
    #         self.stored_scores_dict[old_max_idx] = hi_score
    #         self.stored_clusters[old_max_idx] = clusters[hi_idx]
    #         return self.measure_partition_double_rank(X0)

    #     """
    #         Otherwise the duo is the old_max_idx cluster and the old_second_max_idx cluster
    #     """
    #     second_max_var = np.var(self.stored_clusters[old_second_max_idx], axis=0)
    #     if np.dot(second_max_var, second_max_var) > np.dot(max_var, max_var):
    #         # In this case we act upon the stored second max. We need to replace its data with the hi_score data
    #         X0 = np.copy(self.stored_clusters[old_second_max_idx])
    #         self.stored_scores_dict[old_second_max_idx] = hi_score
    #         self.stored_clusters[old_second_max_idx] = clusters[hi_idx]
    #         return self.measure_partition_double_rank(X0)
        
    #     # But if we act on the stored_max then this is the one we need to replace with hi_score
    #     X0 = np.copy(self.stored_clusters[old_max_idx])
    #     self.stored_scores_dict[old_max_idx] = hi_score
    #     self.stored_clusters[old_max_idx] = clusters[hi_idx]
    #     return self.measure_partition_double_rank(X0)



        # Store the data under the appropriate key
        # parent_id = self.tree.nodes[branch].identifier
        # branch = parent_id + "0"
        # self.tree.create_node(identifier=branch, parent_id=parent_id, data=Patch(inds_0, r0, N0))
        # branch = parent_id + "1"
        # self.tree.create_node(identifier=branch, parent_id=parent_id, data=Patch(inds_1, r1, N1))        
        
        # branch = "0"
        # parent_id = self.tree.nodes[branch].identifier
        # branch = parent_id + "0"
        # self.tree.create_node(identifier=branch, parent_id=parent_id, data=Patch(inds_0[:13], 0.5 * r0, 13))
        # branch = parent_id + "1"
        # self.tree.create_node(identifier=branch, parent_id=parent_id, data=Patch(inds_0[13:], 0.5 * r1, N0 - 13))

        # for parent_node in self.tree.all_nodes_itr():
        #     print("Tag = ", parent_node.tag)
        #     print("Indices = ", parent_node.data.indices)
        #     print("Radius = ", parent_node.data.radius)
        #     print("Sample size = ", parent_node.data.sample_size)
        #     print("")
        # fig, ax = plt.subplots()
        # plt.show()


###### NEW APPROACH PLOTTING STUFF

        # cluster_1_1, cluster_1_2 = self._ball_tree_svm(cluster_1)
        # cluster_2_1, cluster_2_2 = self._ball_tree_svm(cluster_2)

        # fig, axs = plt.subplots(ncols=3, figsize=(18, 6))
        # axs[0].scatter(cluster_1[:, 0], cluster_1[:, 1], s=7, color="aqua")
        # axs[0].scatter(cluster_2[:, 0], cluster_2[:, 1], s=7, color="magenta")

        # axs[1].scatter(cluster_1_1[:, 0], cluster_1_1[:, 1], s=7, color="aqua")
        # axs[1].scatter(cluster_1_2[:, 0], cluster_1_2[:, 1], s=7, color="limegreen")
        # axs[1].scatter(cluster_2[:, 0], cluster_2[:, 1], s=7, color="magenta")

        # axs[2].scatter(cluster_1_1[:, 0], cluster_1_1[:, 1], s=7, color="aqua")
        # axs[2].scatter(cluster_1_2[:, 0], cluster_1_2[:, 1], s=7, color="limegreen")
        # axs[2].scatter(cluster_2_1[:, 0], cluster_2_1[:, 1], s=7, color="magenta")
        # axs[2].scatter(cluster_2_2[:, 0], cluster_2_2[:, 1], s=7, color="orangered")

        # axs[0].set_aspect("equal")
        # axs[1].set_aspect("equal")
        # axs[2].set_aspect("equal")
        # plt.show()



        # cluster_1 = np.append(cluster_1, X_res[assigned_classes == 0], axis=0)
        # cluster_2 = np.append(cluster_2, X_res[assigned_classes == 1], axis=0)



            # We either store the node as an output or split it again. In either case we remove its score and ratio value to prevent them being counted again.
            # self.stored_scores_dict.pop(key)
            # self.stored_rads_dict.pop(key)
            # bm_inds = self.tree.nodes[key].data.indices
            # if len(bm_inds) >= 2:

            #     # We are replacing this node as an output with its two children. Therefore, remove it from the output nodes list.
            #     self.output_nodes.remove(key)

            #     # Split each of the violating clusters and process the diagnostics of the output subclusters
            #     if len(bm_inds) > 2:             
            #         inds, rads, scores = self._ball_tree_svm(self.big_measure[bm_inds])
            #     else:
            #         inds, rads, scores = self._ball_tree_svm(self.big_measure[bm_inds], two_points=True)
            #     self._process_partition_data(self.tree.nodes[key], inds, rads, scores)
            #     if self.cluster_counter == self.n_clusters:
            #         return self.tree, self.output_nodes



        # # Classify any residual data using the SVM
        # if N0 + N1 < n_samples:
        #     mask = np.ones(n_samples, dtype=bool)
        #     mask[inds_0], mask[inds_1] = False, False
        #     res_inds = np.arange(n_samples, dtype=int)[mask]
        #     X_res = X[res_inds]

        #     dist_mask = norm(X_res - X0, ord=self.p, axis=1) < norm(X_res - X1, ord=self.p, axis=1)
        #     X0_assigned_inds = res_inds[dist_mask]
        #     X1_assigned_inds = res_inds[~dist_mask]
        #     inds_0 = np.append(inds_0, X0_assigned_inds)
        #     inds_1 = np.append(inds_1, X1_assigned_inds)
        #     N0 += len(X0_assigned_inds)
        #     N1 += len(X1_assigned_inds)

        #     # training_data = X[np.concatenate([inds_0, inds_1])]
        #     # labels = np.zeros(N0 + N1, dtype=int)
        #     # labels[N0:] = 1
        #     # self.clf.fit(training_data, labels)
        #     # assigned_classes = self.clf.predict(X_res)
        #     # X0_assigned_inds = res_inds[assigned_classes == 0]
        #     # X1_assigned_inds = res_inds[assigned_classes == 1]
        #     # inds_0 = np.append(inds_0, X0_assigned_inds)
        #     # inds_1 = np.append(inds_1, X1_assigned_inds)

        #     # Compute the generalised radii
        #     if len(X0_assigned_inds) > 0:
        #         r0 = np.max(norm(X0 - X[X0_assigned_inds], ord=self.p, axis=1))
        #     else:
        #         r0 = np.max(norm(X0 - X[inds_0], ord=self.p, axis=1))
        #     if len(X1_assigned_inds) > 0:
        #         r1 = np.max(norm(X1 - X[X1_assigned_inds], ord=self.p, axis=1))
        #     else:
        #         r1 = np.max(norm(X1 - X[inds_1], ord=self.p, axis=1))
        #     return [inds_0, inds_1], [r0, r1], [N0, N1]