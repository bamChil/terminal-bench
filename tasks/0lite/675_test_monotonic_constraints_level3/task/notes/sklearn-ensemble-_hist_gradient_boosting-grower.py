"""
This module contains the TreeGrower class.

TreeGrower builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from heapq import heappop, heappush
from timeit import default_timer as time

import numpy as np

from sklearn.ensemble._hist_gradient_boosting._bitset import (
    set_raw_bitset_from_binned_bitset,
)
from sklearn.ensemble._hist_gradient_boosting.common import (
    PREDICTOR_RECORD_DTYPE,
    X_BITSET_INNER_DTYPE,
    MonotonicConstraint,
)
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.ensemble._hist_gradient_boosting.splitting import Splitter
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads




class TreeGrower:
    """
    Tree grower class used to build a tree.

        The tree is fitted to predict the values of a Newton-Raphson step. The
        splits are considered in a best-first fashion, and the quality of a
        split is defined in splitting._split_gain.

        Parameters
        ----------
        X_binned : ndarray of shape (n_samples, n_features), dtype=np.uint8
            The binned input samples. Must be Fortran-aligned.
        gradients : ndarray of shape (n_samples,)
            The gradients of each training sample. Those are the gradients of the
            loss w.r.t the predictions, evaluated at iteration ``i - 1``.
        hessians : ndarray of shape (n_samples,)
            The hessians of each training sample. Those are the hessians of the
            loss w.r.t the predictions, evaluated at iteration ``i - 1``.
        max_leaf_nodes : int, default=None
            The maximum number of leaves for each tree. If None, there is no
            maximum limit.
        max_depth : int, default=None
            The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf.
            Depth isn't constrained by default.
        min_samples_leaf : int, default=20
            The minimum number of samples per leaf.
        min_gain_to_split : float, default=0.
            The minimum gain needed to split a node. Splits with lower gain will
            be ignored.
        min_hessian_to_split : float, default=1e-3
            The minimum sum of hessians needed in each node. Splits that result in
            at least one child having a sum of hessians less than
            ``min_hessian_to_split`` are discarded.
        n_bins : int, default=256
            The total number of bins, including the bin for missing values. Used
            to define the shape of the histograms.
        n_bins_non_missing : ndarray, dtype=np.uint32, default=None
            For each feature, gives the number of bins actually used for
            non-missing values. For features with a lot of unique values, this
            is equal to ``n_bins - 1``. If it's an int, all features are
            considered to have the same number of bins. If None, all features
            are considered to have ``n_bins - 1`` bins.
        has_missing_values : bool or ndarray, dtype=bool, default=False
            Whether each feature contains missing values (in the training data).
            If it's a bool, the same value is used for all features.
        is_categorical : ndarray of bool of shape (n_features,), default=None
            Indicates categorical features.
        monotonic_cst : array-like of int of shape (n_features,), dtype=int, default=None
            Indicates the monotonic constraint to enforce on each feature.
              - 1: monotonic increase
              - 0: no constraint
              - -1: monotonic decrease

            Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
        interaction_cst : list of sets of integers, default=None
            List of interaction constraints.
        l2_regularization : float, default=0.
            The L2 regularization parameter penalizing leaves with small hessians.
            Use ``0`` for no regularization (default).
        feature_fraction_per_split : float, default=1
            Proportion of randomly chosen features in each and every node split.
            This is a form of regularization, smaller values make the trees weaker
            learners and might prevent overfitting.
        rng : Generator
            Numpy random Generator used for feature subsampling.
        shrinkage : float, default=1.
            The shrinkage parameter to apply to the leaves values, also known as
            learning rate.
        n_threads : int, default=None
            Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
            to determine the effective number of threads use, which takes cgroups CPU
            quotes into account. See the docstring of `_openmp_effective_n_threads`
            for details.

        Attributes
        ----------
        histogram_builder : HistogramBuilder
        splitter : Splitter
        root : TreeNode
        finalized_leaves : list of TreeNode
        splittable_nodes : list of TreeNode
        missing_values_bin_idx : int
            Equals n_bins - 1
        n_categorical_splits : int
        n_features : int
        n_nodes : int
        total_find_split_time : float
            Time spent finding the best splits
        total_compute_hist_time : float
            Time spent computing histograms
        total_apply_split_time : float
            Time spent splitting nodes
        with_monotonic_cst : bool
            Whether there are monotonic constraints that apply. False iff monotonic_cst is
            None.

    """

    def __init__(
            self,
            X_binned,
            gradients,
            hessians,
            max_leaf_nodes = None,
            max_depth = None,
            min_samples_leaf = 20,
            min_gain_to_split = 0.0,
            min_hessian_to_split = 0.001,
            n_bins = 256,
            n_bins_non_missing = None,
            has_missing_values = False,
            is_categorical = None,
            monotonic_cst = None,
            interaction_cst = None,
            l2_regularization = 0.0,
            feature_fraction_per_split = 1.0,
            rng = np.random.default_rng(),
            shrinkage = 1.0,
            n_threads = None
        ):
        raise NotImplementedError('This function has been masked for testing')

    def _validate_parameters(
            self,
            X_binned,
            min_gain_to_split,
            min_hessian_to_split
        ):
        """
        Validate parameters passed to __init__.

                Also validate parameters passed to splitter.

        """
        raise NotImplementedError('This function has been masked for testing')

    def grow(self):
        """
        Grow the tree, from root to leaves.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _apply_shrinkage(self):
        """
        Multiply leaves values by shrinkage parameter.

                This must be done at the very end of the growing process. If this were
                done during the growing process e.g. in finalize_leaf(), then a leaf
                would be shrunk but its sibling would potentially not be (if it's a
                non-leaf), which would lead to a wrong computation of the 'middle'
                value needed to enforce the monotonic constraints.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _initialize_root(self):
        """
        Initialize root node and finalize it if needed.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _compute_best_split_and_push(self, node):
        """
        Compute the best possible split (SplitInfo) of a given node.

                Also push it in the heap of splittable nodes if gain isn't zero.
                The gain of a node is 0 if either all the leaves are pure
                (best gain = 0), or if no split would satisfy the constraints,
                (min_hessians_to_split, min_gain_to_split, min_samples_leaf)

        """
        raise NotImplementedError('This function has been masked for testing')

    def split_next(self):
        """
        Split the node with highest potential gain.

                Returns
                -------
                left : TreeNode
                    The resulting left child.
                right : TreeNode
                    The resulting right child.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _compute_interactions(self, node):
        """
        Compute features allowed by interactions to be inherited by child nodes.

                Example: Assume constraints [{0, 1}, {1, 2}].
                   1      <- Both constraint groups could be applied from now on
                  / \
                 1   2    <- Left split still fulfills both constraint groups.
                / \ / \      Right split at feature 2 has only group {1, 2} from now on.

                LightGBM uses the same logic for overlapping groups. See
                https://github.com/microsoft/LightGBM/issues/4481 for details.

                Parameters:
                ----------
                node : TreeNode
                    A node that might have children. Based on its feature_idx, the interaction
                    constraints for possible child nodes are computed.

                Returns
                -------
                allowed_features : ndarray, dtype=uint32
                    Indices of features allowed to split for children.
                interaction_cst_indices : list of ints
                    Indices of the interaction sets that have to be applied on splits of
                    child nodes. The fewer sets the stronger the constraint as fewer sets
                    contain fewer features.

        """
        raise NotImplementedError('This function has been masked for testing')

    def _finalize_leaf(self, node):
        """
        Make node a leaf of the tree being grown.
        """
        raise NotImplementedError('This function has been masked for testing')

    def _finalize_splittable_nodes(self):
        """
        Transform all splittable nodes into leaves.

                Used when some constraint is met e.g. maximum number of leaves or
                maximum depth.
        """
        raise NotImplementedError('This function has been masked for testing')

    def make_predictor(self, binning_thresholds):
        """
        Make a TreePredictor object out of the current tree.

                Parameters
                ----------
                binning_thresholds : array-like of floats
                    Corresponds to the bin_thresholds_ attribute of the BinMapper.
                    For each feature, this stores:

                    - the bin frontiers for continuous features
                    - the unique raw category values for categorical features

                Returns
                -------
                A TreePredictor object.

        """
        raise NotImplementedError('This function has been masked for testing')

