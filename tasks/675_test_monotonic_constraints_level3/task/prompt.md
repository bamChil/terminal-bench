## Task
## Task: Implement Histogram-Based Gradient Boosting Models

**Core Functionality:**
Build fast gradient boosting estimators for regression and classification using histogram-based tree growing, optimized for large datasets (10,000+ samples) with native missing value support.

**Main Features & Requirements:**
- **Dual Models**: Implement both regressor and classifier with configurable loss functions (squared_error, log_loss, etc.)
- **Efficient Tree Growing**: Use histogram-based splitting with binned features for speed optimization
- **Missing Value Handling**: Native support for NaN values with learned split directions
- **Advanced Constraints**: Support monotonic constraints, interaction constraints, and categorical features
- **Early Stopping**: Implement validation-based early stopping with configurable scoring metrics
- **Prediction Methods**: Provide standard prediction plus staged prediction for monitoring training progress

**Key Challenges:**
- **Performance Optimization**: Balance speed gains from binning while maintaining prediction accuracy
- **Constraint Enforcement**: Properly handle monotonic and interaction constraints during tree construction
- **Memory Management**: Efficiently manage histogram computations and tree structures for large datasets
- **Parameter Validation**: Ensure robust parameter validation across numerous hyperparameters and constraint combinations
- **Multi-class Handling**: Scale appropriately from binary to multi-class classification (multiple trees per iteration)

**NOTE**: 
- This test is derived from the `scikit-learn` library, but you are NOT allowed to view this codebase or call any of its interfaces. It is **VERY IMPORTANT** to note that if we detect any viewing or calling of this codebase, you will receive a ZERO for this review.
- What's more, you need to install `pytest, pytest-timeout, pytest-json-report` in your environment, otherwise our tests won't run and you'll get **ZERO POINTS**!
- **CRITICAL**: This task is derived from `scikit-learn`, but you **MUST** implement the task description independently. It is **ABSOLUTELY FORBIDDEN** to use `pip install scikit-learn` or some similar commands to access the original implementation—doing so will be considered cheating and will result in an immediate score of ZERO! You must keep this firmly in mind throughout your implementation.
- You are now in `/testbed/`, and originally there was a specific implementation of `scikit-learn` under `/testbed/` that had been installed via `pip install -e .`. However, to prevent you from cheating, we've removed the code under `/testbed/`. While you can see traces of the installation via the pip show, it's an artifact, and `scikit-learn` doesn't exist. So you can't and don't need to use `pip install scikit-learn`, just focus on writing your `agent_code` and accomplishing our task.
- Also, don't try to `pip uninstall scikit-learn` even if the actual `scikit-learn` has already been deleted by us, as this will affect our evaluation of you, and uninstalling the residual `scikit-learn` will result in you getting a ZERO because our tests won't run.

Your available resources are listed below:
- `/testbed/ace_bench/task/black_links.txt`: Prohibited URLs (all other web resources are allowed)


## Precautions
- You may need to install some of the libraries to support you in accomplishing our task, some of the packages are already pre-installed in your environment, you can check them out yourself via `pip list` etc. For standard installs, just run `pip install <package>`. There's no need to add `--index-url`, the domestic mirrors are already set up unless you have special requirements.
- Please note that when running `pip install <package>`, you should not include the `--force-reinstall` flag, as it may cause pre-installed packages to be reinstalled.
- **IMPORTANT**: While you can install libraries using pip, you should never access the actual implementations in the libraries you install, as the tasks we give you originate from github, and if you look at the contents of the libraries, it could result in you being awarded 0 points directly for alleged cheating. Specifically, you cannot read any files under `/usr/local/lib/python3.x` and its subfolders (here python3.x means any version of python).
- **IMPORTANT**: Your installed python library may contain a real implementation of the task, and you are prohibited from directly calling the library's interface of the same name and pretending to package it as your answer, which will also be detected and awarded 0 points.
- **CRITICAL REQUIREMENT**: After completing the task, pytest will be used to test your implementation. **YOU MUST**:
    - Build proper code hierarchy with correct import relationships shown in **Test Description** (I will give you this later)
    - Match the exact interface shown in the **Interface Description** (I will give you this later)
- I will tell you details about **CRITICAL REQUIREMENT** below.

Your final deliverable should be code in the `/testbed/agent_code` directory.
The final structure is like below, note that all dirs and files under agent_code/ are just examples, your codebase's structure should match import structure in **Test Description**, which I will tell you later.
```
/testbed
├── agent_code/           # all your code should be put into this dir and match the specific dir structure
│   ├── __init__.py       # agent_code/ folder and ALL folders under it should contain __init__.py
│   ├── dir1/
│   │   ├── __init__.py
│   │   ├── code1.py
│   │   ├── ...
├── setup.py              # after finishing your work, you MUST generate this file
```
After you have done all your work, you need to complete three CRITICAL things: 
1. When you have generated all files or folders under `agent_code/` that match the directory structure, you need to recursively generate `__init__.py` under `agent_code/` and in all subfolders under it to ensure that we can access all functions you generate.(you can simply generate empty `__init__.py`)
2. You need to generate `/testbed/setup.py` under `/testbed/` and place the following content exactly:
```python
from setuptools import setup, find_packages
setup(
    name="agent_code",
    version="1.0.0",
    packages=find_packages(include=["agent_code", "agent_code.*"]),
    install_requires=[],
)
```
3. After you have done above two things, you need to use `cd /testbed && pip install .` command to install your code.
Remember, these things are **VERY IMPORTANT**, as they will directly affect whether you can pass our tests.

## Test and Interface Descriptions

The **Test Description** will tell you the position of the function or class which we're testing should satisfy.
This means that when you generate some files and complete the functionality we want to test in the files, you need to put these files in the specified directory, otherwise our tests won't be able to import your generated.
For example, if the **Test Description** show you this:
```python
from agent_code.sklearn.ensemble import HistGradientBoostingClassifier
```
This means that we will test one function/class: HistGradientBoostingClassifier.
And the defination and implementation of class HistGradientBoostingClassifier should be in `/testbed/agent_code/sklearn/ensemble.py`. And the same applies to others.

In addition to the above path requirements, you may try to modify any file in codebase that you feel will help you accomplish our task. However, please note that you may cause our test to fail if you arbitrarily modify or delete some generic functions in existing files, so please be careful in completing your work.
And note that there may be not only one **Test Description**, you should match all **Test Description {n}** 

The **Interface Description**  describes what the functions we are testing do and the input and output formats.
for example, you will get things like this:
```python
class HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting):
    """
    Histogram-based Gradient Boosting Classification Tree.
    
        This estimator is much faster than
        :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
        for big datasets (n_samples >= 10 000).
    
        This estimator has native support for missing values (NaNs). During
        training, the tree grower learns at each split point whether samples
        with missing values should go to the left or right child, based on the
        potential gain. When predicting, samples with missing values are
        assigned to the left or right child consequently. If no missing values
        were encountered for a given feature during training, then samples with
        missing values are mapped to whichever child has the most samples.
    
        This implementation is inspired by
        `LightGBM <https://github.com/Microsoft/LightGBM>`_.
    
        Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.
    
        .. versionadded:: 0.21
    
        Parameters
        ----------
        loss : {'log_loss'}, default='log_loss'
            The loss function to use in the boosting process.
    
            For binary classification problems, 'log_loss' is also known as logistic loss,
            binomial deviance or binary crossentropy. Internally, the model fits one tree
            per boosting iteration and uses the logistic sigmoid function (expit) as
            inverse link function to compute the predicted positive class probability.
    
            For multiclass classification problems, 'log_loss' is also known as multinomial
            deviance or categorical crossentropy. Internally, the model fits one tree per
            boosting iteration and per class and uses the softmax function as inverse link
            function to compute the predicted probabilities of the classes.
    
        learning_rate : float, default=0.1
            The learning rate, also known as *shrinkage*. This is used as a
            multiplicative factor for the leaves values. Use ``1`` for no
            shrinkage.
        max_iter : int, default=100
            The maximum number of iterations of the boosting process, i.e. the
            maximum number of trees for binary classification. For multiclass
            classification, `n_classes` trees per iteration are built.
        max_leaf_nodes : int or None, default=31
            The maximum number of leaves for each tree. Must be strictly greater
            than 1. If None, there is no maximum limit.
        max_depth : int or None, default=None
            The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf.
            Depth isn't constrained by default.
        min_samples_leaf : int, default=20
            The minimum number of samples per leaf. For small datasets with less
            than a few hundred samples, it is recommended to lower this value
            since only very shallow trees would be built.
        l2_regularization : float, default=0
            The L2 regularization parameter penalizing leaves with small hessians.
            Use ``0`` for no regularization (default).
        max_features : float, default=1.0
            Proportion of randomly chosen features in each and every node split.
            This is a form of regularization, smaller values make the trees weaker
            learners and might prevent overfitting.
            If interaction constraints from `interaction_cst` are present, only allowed
            features are taken into account for the subsampling.
    
            .. versionadded:: 1.4
    
        max_bins : int, default=255
            The maximum number of bins to use for non-missing values. Before
            training, each feature of the input array `X` is binned into
            integer-valued bins, which allows for a much faster training stage.
            Features with a small number of unique values may use less than
            ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
            is always reserved for missing values. Must be no larger than 255.
        categorical_features : array-like of {bool, int, str} of shape (n_features)             or shape (n_categorical_features,), default='from_dtype'
            Indicates the categorical features.
    
            - None : no feature will be considered categorical.
            - boolean array-like : boolean mask indicating categorical features.
            - integer array-like : integer indices indicating categorical
              features.
            - str array-like: names of categorical features (assuming the training
              data has feature names).
            - `"from_dtype"`: dataframe columns with dtype "category" are
              considered to be categorical features. The input must be an object
              exposing a ``__dataframe__`` method such as pandas or polars
              DataFrames to use this feature.
    
            For each categorical feature, there must be at most `max_bins` unique
            categories. Negative values for categorical features encoded as numeric
            dtypes are treated as missing values. All categorical values are
            converted to floating point numbers. This means that categorical values
            of 1.0 and 1 are treated as the same category.
    
            Read more in the :ref:`User Guide <categorical_support_gbdt>`.
    
            .. versionadded:: 0.24
    
            .. versionchanged:: 1.2
               Added support for feature names.
    
            .. versionchanged:: 1.4
               Added `"from_dtype"` option.
    
            .. versionchanged:: 1.6
               The default value changed from `None` to `"from_dtype"`.
    
        monotonic_cst : array-like of int of shape (n_features) or dict, default=None
            Monotonic constraint to enforce on each feature are specified using the
            following integer values:
    
            - 1: monotonic increase
            - 0: no constraint
            - -1: monotonic decrease
    
            If a dict with str keys, map feature to monotonic constraints by name.
            If an array, the features are mapped to constraints by position. See
            :ref:`monotonic_cst_features_names` for a usage example.
    
            The constraints are only valid for binary classifications and hold
            over the probability of the positive class.
            Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    
            .. versionadded:: 0.23
    
            .. versionchanged:: 1.2
               Accept dict of constraints with feature names as keys.
    
        interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets             of int, default=None
            Specify interaction constraints, the sets of features which can
            interact with each other in child node splits.
    
            Each item specifies the set of feature indices that are allowed
            to interact with each other. If there are more features than
            specified in these constraints, they are treated as if they were
            specified as an additional set.
    
            The strings "pairwise" and "no_interactions" are shorthands for
            allowing only pairwise or no interactions, respectively.
    
            For instance, with 5 features in total, `interaction_cst=[{0, 1}]`
            is equivalent to `interaction_cst=[{0, 1}, {2, 3, 4}]`,
            and specifies that each branch of a tree will either only split
            on features 0 and 1 or only split on features 2, 3 and 4.
    
            See :ref:`this example<ice-vs-pdp>` on how to use `interaction_cst`.
    
            .. versionadded:: 1.2
    
        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble. For results to be valid, the
            estimator should be re-trained on the same data only.
            See :term:`the Glossary <warm_start>`.
        early_stopping : 'auto' or bool, default='auto'
            If 'auto', early stopping is enabled if the sample size is larger than
            10000 or if `X_val` and `y_val` are passed to `fit`. If True, early stopping
            is enabled, otherwise early stopping is disabled.
    
            .. versionadded:: 0.23
    
        scoring : str or callable or None, default='loss'
            Scoring method to use for early stopping. Only used if `early_stopping`
            is enabled. Options:
    
            - str: see :ref:`scoring_string_names` for options.
            - callable: a scorer callable object (e.g., function) with signature
              ``scorer(estimator, X, y)``. See :ref:`scoring_callable` for details.
            - `None`: :ref:`accuracy <accuracy_score>` is used.
            - 'loss': early stopping is checked w.r.t the loss value.
    
        validation_fraction : int or float or None, default=0.1
            Proportion (or absolute size) of training data to set aside as
            validation data for early stopping. If None, early stopping is done on
            the training data.
            The value is ignored if either early stopping is not performed, e.g.
            `early_stopping=False`, or if `X_val` and `y_val` are passed to fit.
        n_iter_no_change : int, default=10
            Used to determine when to "early stop". The fitting process is
            stopped when none of the last ``n_iter_no_change`` scores are better
            than the ``n_iter_no_change - 1`` -th-to-last one, up to some
            tolerance. Only used if early stopping is performed.
        tol : float, default=1e-7
            The absolute tolerance to use when comparing scores. The higher the
            tolerance, the more likely we are to early stop: higher tolerance
            means that it will be harder for subsequent iterations to be
            considered an improvement upon the reference score.
        verbose : int, default=0
            The verbosity level. If not zero, print some information about the
            fitting process. ``1`` prints only summary info, ``2`` prints info per
            iteration.
        random_state : int, RandomState instance or None, default=None
            Pseudo-random number generator to control the subsampling in the
            binning process, and the train/validation data split if early stopping
            is enabled.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form `{class_label: weight}`.
            If not given, all classes are supposed to have weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as `n_samples / (n_classes * np.bincount(y))`.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if `sample_weight` is specified.
    
            .. versionadded:: 1.2
    
        Attributes
        ----------
        classes_ : array, shape = (n_classes,)
            Class labels.
        do_early_stopping_ : bool
            Indicates whether early stopping is used during training.
        n_iter_ : int
            The number of iterations as selected by early stopping, depending on
            the `early_stopping` parameter. Otherwise it corresponds to max_iter.
        n_trees_per_iteration_ : int
            The number of tree that are built at each iteration. This is equal to 1
            for binary classification, and to ``n_classes`` for multiclass
            classification.
        train_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the training data. The first entry
            is the score of the ensemble before the first iteration. Scores are
            computed according to the ``scoring`` parameter. If ``scoring`` is
            not 'loss', scores are computed on a subset of at most 10 000
            samples. Empty if no early stopping.
        validation_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the held-out validation data. The
            first entry is the score of the ensemble before the first iteration.
            Scores are computed according to the ``scoring`` parameter. Empty if
            no early stopping or if ``validation_fraction`` is None.
        is_categorical_ : ndarray, shape (n_features, ) or None
            Boolean mask for the categorical features. ``None`` if there are no
            categorical features.
        n_features_in_ : int
            Number of features seen during :term:`fit`.
    
            .. versionadded:: 0.24
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
    
            .. versionadded:: 1.0
    
        See Also
        --------
        GradientBoostingClassifier : Exact gradient boosting method that does not
            scale as good on datasets with a large number of samples.
        sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
        RandomForestClassifier : A meta-estimator that fits a number of decision
            tree classifiers on various sub-samples of the dataset and uses
            averaging to improve the predictive accuracy and control over-fitting.
        AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
            on the original dataset and then fits additional copies of the
            classifier on the same dataset where the weights of incorrectly
            classified instances are adjusted such that subsequent classifiers
            focus more on difficult cases.
    
        Examples
        --------
        >>> from sklearn.ensemble import HistGradientBoostingClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = HistGradientBoostingClassifier().fit(X, y)
        >>> clf.score(X, y)
        1.0
        
    """

    _parameter_constraints = "{**BaseHistGradientBoosting._parameter_constraints, 'loss': [StrOptions({'log_loss'}), BaseLoss], 'class_weight': [dict, StrOptions({'balanced'}), None]}"

    def __init__(self, loss = 'log_loss'):
        """
        Initialize a histogram-based gradient boosting classifier.

        This constructor sets up the basic parameters for the HistGradientBoostingClassifier,
        which implements a fast gradient boosting algorithm using histogram-based tree growing.

        Parameters
        ----------
        loss : {'log_loss'} or BaseLoss instance, default='log_loss'
            The loss function to use in the boosting process. For binary classification,
            'log_loss' uses logistic loss (binomial deviance). For multiclass classification,
            'log_loss' uses multinomial deviance (categorical crossentropy). Custom loss
            functions can be provided as BaseLoss instances.

        learning_rate : float, default=0.1
            The learning rate (shrinkage factor) applied to each tree's contribution.
            Lower values make the model more robust but require more iterations.
            Must be greater than 0.

        max_iter : int, default=100
            Maximum number of boosting iterations. For binary classification, this is
            the number of trees. For multiclass, n_classes trees are built per iteration.
            Must be at least 1.

        max_leaf_nodes : int or None, default=31
            Maximum number of leaves per tree. Must be greater than 1 if specified.
            If None, no limit is imposed on the number of leaves.

        max_depth : int or None, default=None
            Maximum depth of each tree (number of edges from root to deepest leaf).
            If None, depth is not constrained.

        min_samples_leaf : int, default=20
            Minimum number of samples required in each leaf node. For small datasets,
            consider lowering this value to allow deeper trees.

        l2_regularization : float, default=0.0
            L2 regularization parameter for leaf values. Higher values provide more
            regularization. Must be non-negative.

        max_features : float, default=1.0
            Proportion of features to consider at each split (0 < max_features <= 1).
            Provides regularization by introducing randomness in feature selection.

        max_bins : int, default=255
            Maximum number of bins for discretizing continuous features. Must be
            between 2 and 255. One additional bin is reserved for missing values.

        categorical_features : array-like, str or None, default='from_dtype'
            Specification of categorical features. Can be boolean mask, integer indices,
            feature names, or 'from_dtype' to auto-detect from pandas categorical columns.

        monotonic_cst : array-like, dict or None, default=None
            Monotonic constraints for features. Values: 1 (increasing), 0 (no constraint),
            -1 (decreasing). Only supported for binary classification.

        interaction_cst : sequence, str or None, default=None
            Interaction constraints specifying which features can interact in splits.
            Can be 'pairwise', 'no_interactions', or custom feature groupings.

        warm_start : bool, default=False
            Whether to reuse the previous fit solution and add more estimators.
            Requires fitting on the same data.

        early_stopping : 'auto' or bool, default='auto'
            Whether to use early stopping. 'auto' enables it for large datasets
            (n_samples > 10000) or when validation data is provided.

        scoring : str, callable or None, default='loss'
            Scoring method for early stopping. Can be a scorer name, callable,
            None (uses accuracy), or 'loss' (uses the loss function).

        validation_fraction : float, int or None, default=0.1
            Fraction or absolute size of training data for validation in early stopping.
            Ignored if validation data is explicitly provided or early stopping is disabled.

        n_iter_no_change : int, default=10
            Number of iterations with no improvement to trigger early stopping.
            Must be at least 1.

        tol : float, default=1e-7
            Tolerance for early stopping. Higher values make early stopping more likely.
            Must be non-negative.

        verbose : int, default=0
            Verbosity level. 0 for silent, 1 for summary info, 2 for detailed iteration info.

        random_state : int, RandomState or None, default=None
            Random state for reproducible results. Controls binning randomness and
            train/validation splits.

        class_weight : dict, 'balanced' or None, default=None
            Weights for classes. 'balanced' uses inverse class frequencies.
            Custom weights as {class_label: weight} dictionary.

        Notes
        -----
        This initializer only sets the parameters. The actual model training occurs
        when the fit method is called. The classifier supports both binary and
        multiclass classification automatically based on the target variable.

        For binary classification, one tree per iteration is built. For multiclass
        classification, one tree per class per iteration is built.
        """
        <your code>
...
```

In order to implement this functionality, some additional libraries etc. are often required, I don't restrict you to any libraries, you need to think about what dependencies you might need and fetch and install and call them yourself. The only thing is that you **MUST** fulfill the input/output format described by this interface, otherwise the test will not pass and you will get zero points for this feature.
And note that there may be not only one **Interface Description**, you should match all **Interface Description {n}**

### Test Description 1
Below is **Test Description 1**
```python
from agent_code.sklearn.ensemble import HistGradientBoostingClassifier
from agent_code.sklearn.ensemble import HistGradientBoostingRegressor
from agent_code.sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
```

### Interface Description 1
Below is **Interface Description 1** for file: sklearn-ensemble-_hist_gradient_boosting-gradient_boosting.py

This file contains 2 top-level interface(s) that need to be implemented.

```python
class HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting):
    """
    Histogram-based Gradient Boosting Regression Tree.
    
        This estimator is much faster than
        :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
        for big datasets (n_samples >= 10 000).
    
        This estimator has native support for missing values (NaNs). During
        training, the tree grower learns at each split point whether samples
        with missing values should go to the left or right child, based on the
        potential gain. When predicting, samples with missing values are
        assigned to the left or right child consequently. If no missing values
        were encountered for a given feature during training, then samples with
        missing values are mapped to whichever child has the most samples.
        See :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` for a
        usecase example of this feature.
    
        This implementation is inspired by
        `LightGBM <https://github.com/Microsoft/LightGBM>`_.
    
        Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.
    
        .. versionadded:: 0.21
    
        Parameters
        ----------
        loss : {'squared_error', 'absolute_error', 'gamma', 'poisson', 'quantile'},             default='squared_error'
            The loss function to use in the boosting process. Note that the
            "squared error", "gamma" and "poisson" losses actually implement
            "half least squares loss", "half gamma deviance" and "half poisson
            deviance" to simplify the computation of the gradient. Furthermore,
            "gamma" and "poisson" losses internally use a log-link, "gamma"
            requires ``y > 0`` and "poisson" requires ``y >= 0``.
            "quantile" uses the pinball loss.
    
            .. versionchanged:: 0.23
               Added option 'poisson'.
    
            .. versionchanged:: 1.1
               Added option 'quantile'.
    
            .. versionchanged:: 1.3
               Added option 'gamma'.
    
        quantile : float, default=None
            If loss is "quantile", this parameter specifies which quantile to be estimated
            and must be between 0 and 1.
        learning_rate : float, default=0.1
            The learning rate, also known as *shrinkage*. This is used as a
            multiplicative factor for the leaves values. Use ``1`` for no
            shrinkage.
        max_iter : int, default=100
            The maximum number of iterations of the boosting process, i.e. the
            maximum number of trees.
        max_leaf_nodes : int or None, default=31
            The maximum number of leaves for each tree. Must be strictly greater
            than 1. If None, there is no maximum limit.
        max_depth : int or None, default=None
            The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf.
            Depth isn't constrained by default.
        min_samples_leaf : int, default=20
            The minimum number of samples per leaf. For small datasets with less
            than a few hundred samples, it is recommended to lower this value
            since only very shallow trees would be built.
        l2_regularization : float, default=0
            The L2 regularization parameter penalizing leaves with small hessians.
            Use ``0`` for no regularization (default).
        max_features : float, default=1.0
            Proportion of randomly chosen features in each and every node split.
            This is a form of regularization, smaller values make the trees weaker
            learners and might prevent overfitting.
            If interaction constraints from `interaction_cst` are present, only allowed
            features are taken into account for the subsampling.
    
            .. versionadded:: 1.4
    
        max_bins : int, default=255
            The maximum number of bins to use for non-missing values. Before
            training, each feature of the input array `X` is binned into
            integer-valued bins, which allows for a much faster training stage.
            Features with a small number of unique values may use less than
            ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
            is always reserved for missing values. Must be no larger than 255.
        categorical_features : array-like of {bool, int, str} of shape (n_features)             or shape (n_categorical_features,), default='from_dtype'
            Indicates the categorical features.
    
            - None : no feature will be considered categorical.
            - boolean array-like : boolean mask indicating categorical features.
            - integer array-like : integer indices indicating categorical
              features.
            - str array-like: names of categorical features (assuming the training
              data has feature names).
            - `"from_dtype"`: dataframe columns with dtype "category" are
              considered to be categorical features. The input must be an object
              exposing a ``__dataframe__`` method such as pandas or polars
              DataFrames to use this feature.
    
            For each categorical feature, there must be at most `max_bins` unique
            categories. Negative values for categorical features encoded as numeric
            dtypes are treated as missing values. All categorical values are
            converted to floating point numbers. This means that categorical values
            of 1.0 and 1 are treated as the same category.
    
            Read more in the :ref:`User Guide <categorical_support_gbdt>` and
            :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_categorical.py`.
    
            .. versionadded:: 0.24
    
            .. versionchanged:: 1.2
               Added support for feature names.
    
            .. versionchanged:: 1.4
               Added `"from_dtype"` option.
    
            .. versionchanged:: 1.6
               The default value changed from `None` to `"from_dtype"`.
    
        monotonic_cst : array-like of int of shape (n_features) or dict, default=None
            Monotonic constraint to enforce on each feature are specified using the
            following integer values:
    
            - 1: monotonic increase
            - 0: no constraint
            - -1: monotonic decrease
    
            If a dict with str keys, map feature to monotonic constraints by name.
            If an array, the features are mapped to constraints by position. See
            :ref:`monotonic_cst_features_names` for a usage example.
    
            Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    
            .. versionadded:: 0.23
    
            .. versionchanged:: 1.2
               Accept dict of constraints with feature names as keys.
    
        interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets             of int, default=None
            Specify interaction constraints, the sets of features which can
            interact with each other in child node splits.
    
            Each item specifies the set of feature indices that are allowed
            to interact with each other. If there are more features than
            specified in these constraints, they are treated as if they were
            specified as an additional set.
    
            The strings "pairwise" and "no_interactions" are shorthands for
            allowing only pairwise or no interactions, respectively.
    
            For instance, with 5 features in total, `interaction_cst=[{0, 1}]`
            is equivalent to `interaction_cst=[{0, 1}, {2, 3, 4}]`,
            and specifies that each branch of a tree will either only split
            on features 0 and 1 or only split on features 2, 3 and 4.
    
            See :ref:`this example<ice-vs-pdp>` on how to use `interaction_cst`.
    
            .. versionadded:: 1.2
    
        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble. For results to be valid, the
            estimator should be re-trained on the same data only.
            See :term:`the Glossary <warm_start>`.
        early_stopping : 'auto' or bool, default='auto'
            If 'auto', early stopping is enabled if the sample size is larger than
            10000 or if `X_val` and `y_val` are passed to `fit`. If True, early stopping
            is enabled, otherwise early stopping is disabled.
    
            .. versionadded:: 0.23
    
        scoring : str or callable or None, default='loss'
            Scoring method to use for early stopping. Only used if `early_stopping`
            is enabled. Options:
    
            - str: see :ref:`scoring_string_names` for options.
            - callable: a scorer callable object (e.g., function) with signature
              ``scorer(estimator, X, y)``. See :ref:`scoring_callable` for details.
            - `None`: the :ref:`coefficient of determination <r2_score>`
              (:math:`R^2`) is used.
            - 'loss': early stopping is checked w.r.t the loss value.
    
        validation_fraction : int or float or None, default=0.1
            Proportion (or absolute size) of training data to set aside as
            validation data for early stopping. If None, early stopping is done on
            the training data.
            The value is ignored if either early stopping is not performed, e.g.
            `early_stopping=False`, or if `X_val` and `y_val` are passed to fit.
        n_iter_no_change : int, default=10
            Used to determine when to "early stop". The fitting process is
            stopped when none of the last ``n_iter_no_change`` scores are better
            than the ``n_iter_no_change - 1`` -th-to-last one, up to some
            tolerance. Only used if early stopping is performed.
        tol : float, default=1e-7
            The absolute tolerance to use when comparing scores during early
            stopping. The higher the tolerance, the more likely we are to early
            stop: higher tolerance means that it will be harder for subsequent
            iterations to be considered an improvement upon the reference score.
        verbose : int, default=0
            The verbosity level. If not zero, print some information about the
            fitting process. ``1`` prints only summary info, ``2`` prints info per
            iteration.
        random_state : int, RandomState instance or None, default=None
            Pseudo-random number generator to control the subsampling in the
            binning process, and the train/validation data split if early stopping
            is enabled.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
    
        Attributes
        ----------
        do_early_stopping_ : bool
            Indicates whether early stopping is used during training.
        n_iter_ : int
            The number of iterations as selected by early stopping, depending on
            the `early_stopping` parameter. Otherwise it corresponds to max_iter.
        n_trees_per_iteration_ : int
            The number of tree that are built at each iteration. For regressors,
            this is always 1.
        train_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the training data. The first entry
            is the score of the ensemble before the first iteration. Scores are
            computed according to the ``scoring`` parameter. If ``scoring`` is
            not 'loss', scores are computed on a subset of at most 10 000
            samples. Empty if no early stopping.
        validation_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the held-out validation data. The
            first entry is the score of the ensemble before the first iteration.
            Scores are computed according to the ``scoring`` parameter. Empty if
            no early stopping or if ``validation_fraction`` is None.
        is_categorical_ : ndarray, shape (n_features, ) or None
            Boolean mask for the categorical features. ``None`` if there are no
            categorical features.
        n_features_in_ : int
            Number of features seen during :term:`fit`.
    
            .. versionadded:: 0.24
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
    
            .. versionadded:: 1.0
    
        See Also
        --------
        GradientBoostingRegressor : Exact gradient boosting method that does not
            scale as good on datasets with a large number of samples.
        sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
        RandomForestRegressor : A meta-estimator that fits a number of decision
            tree regressors on various sub-samples of the dataset and uses
            averaging to improve the statistical performance and control
            over-fitting.
        AdaBoostRegressor : A meta-estimator that begins by fitting a regressor
            on the original dataset and then fits additional copies of the
            regressor on the same dataset but where the weights of instances are
            adjusted according to the error of the current prediction. As such,
            subsequent regressors focus more on difficult cases.
    
        Examples
        --------
        >>> from sklearn.ensemble import HistGradientBoostingRegressor
        >>> from sklearn.datasets import load_diabetes
        >>> X, y = load_diabetes(return_X_y=True)
        >>> est = HistGradientBoostingRegressor().fit(X, y)
        >>> est.score(X, y)
        0.92...
        
    """

    _parameter_constraints = "{**BaseHistGradientBoosting._parameter_constraints, 'loss': [StrOptions({'squared_error', 'absolute_error', 'poisson', 'gamma', 'quantile'}), BaseLoss], 'quantile': [Interval(Real, 0, 1, closed='both'), None]}"

    def __init__(
        self,
        loss = 'squared_error'
    ):
        """
        Initialize a histogram-based gradient boosting estimator.

        This constructor sets up the base parameters for histogram-based gradient boosting
        models used in both regression and classification tasks.

        Parameters
        ----------
        loss : str or BaseLoss instance, default='squared_error'
            The loss function to use in the boosting process. For regression tasks,
            options include 'squared_error', 'absolute_error', 'gamma', 'poisson', 
            and 'quantile'. For classification tasks, 'log_loss' is typically used.
            Can also be a custom BaseLoss instance for advanced use cases.

        Notes
        -----
        This is an abstract base class constructor that should not be called directly.
        Use HistGradientBoostingRegressor or HistGradientBoostingClassifier instead.

        The loss parameter is validated during the fitting process, and the actual
        loss object is created based on the specific requirements of the task
        (regression vs classification) and any additional parameters like sample
        weights or quantile values.
        """
        <your code>

    def predict(self, X):
        """
        Predict values for X.

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.

                Returns
                -------
                y : ndarray, shape (n_samples,)
                    The predicted values.

        """
        <your code>

    def staged_predict(self, X):
        """
        Predict regression target for each iteration.

                This method allows monitoring (i.e. determine error on testing set)
                after each stage.

                .. versionadded:: 0.24

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The input samples.

                Yields
                ------
                y : generator of ndarray of shape (n_samples,)
                    The predicted values of the input samples, for each iteration.

        """
        <your code>

    def _encode_y(self, y):
       
        """
        Encode target values for training.

        This method processes and validates the target values `y` to prepare them for
        the gradient boosting training process. It performs necessary transformations
        and sets up class-related attributes for the estimator.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values to be encoded. For classification tasks, these are the
            class labels. For regression tasks, these are the continuous target
            values.

        Returns
        -------
        encoded_y : ndarray of shape (n_samples,)
            The encoded target values. For regression, this is typically the input
            array converted to the appropriate dtype. For classification, this
            contains the label-encoded class indices.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses. The
        specific encoding behavior depends on whether the estimator is used for
        classification or regression:

        - For regression: Converts y to the expected dtype and validates constraints
          specific to certain loss functions (e.g., gamma loss requires y > 0)
        - For classification: Uses a LabelEncoder to convert class labels to integer
          indices and sets up the classes_ attribute

        The method also sets the n_trees_per_iteration_ attribute, which determines
        how many trees are built at each boosting iteration (1 for regression and
        binary classification, n_classes for multiclass classification).

        This method is called during the fit process and is essential for preparing
        the target values in the format expected by the gradient boosting algorithm.
        """
    
        <your code>

    def _encode_y_val(self, y = None):
        """
        Encode validation target values for internal use during fitting.

        This method transforms validation target values (y_val) to the same encoded format
        used internally by the gradient boosting algorithm. For regressors, this typically
        involves dtype conversion and validation. For classifiers, this involves label
        encoding to convert class labels to integer indices.

        Parameters
        ----------
        y : array-like of shape (n_samples,), default=None
            The validation target values to encode. If None, no encoding is performed.
            For regressors, these should be continuous values. For classifiers, these
            should be class labels that were seen during the initial fit.

        Returns
        -------
        encoded_y : ndarray of shape (n_samples,) or None
            The encoded validation target values with appropriate dtype for internal
            computations. Returns None if input y is None. For regressors, returns
            y converted to Y_DTYPE. For classifiers, returns integer-encoded labels
            using the same label encoder fitted during training.

        Notes
        -----
        This method is called internally during fit when validation data is provided
        via X_val and y_val parameters. It ensures that validation targets are in
        the same format as training targets for consistent evaluation during early
        stopping.

        For classifiers, the validation labels must contain only classes that were
        seen during training, otherwise a ValueError will be raised by the internal
        label encoder.

        For regressors with specific loss functions (e.g., 'gamma', 'poisson'), the
        same constraints that apply to training targets also apply to validation
        targets.
        """
        <your code>

    def _get_loss(self, sample_weight):
        """
        Get the appropriate loss function for the gradient boosting model.

        This method creates and returns a loss function instance based on the model's
        loss parameter and current configuration. The loss function is used throughout
        the boosting process to compute gradients and hessians for tree fitting.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,) or None
            Sample weights for the training data. If None, uniform weights are assumed.
            The loss function uses these weights to properly weight the contribution of
            each sample during gradient and hessian computation.

        Returns
        -------
        loss : BaseLoss
            An instance of the appropriate loss function class. The specific type depends
            on the model's loss parameter:

            For HistGradientBoostingRegressor:
            - Returns PinballLoss for quantile regression (requires quantile parameter)
            - Returns corresponding loss class for other regression losses

            For HistGradientBoostingClassifier:
            - Returns HalfBinomialLoss for binary classification
            - Returns HalfMultinomialLoss for multiclass classification

        Notes
        -----
        This is an abstract method that must be implemented by subclasses. The method
        is called during the fit process to initialize the loss function with the
        appropriate sample weights.

        The loss function instance is used to:
        - Compute initial baseline predictions
        - Calculate gradients and hessians at each boosting iteration
        - Evaluate training and validation scores during early stopping
        - Transform raw predictions to final predictions

        For classification tasks, the number of trees per iteration (n_trees_per_iteration_)
        must be set before calling this method, as it determines whether to use binomial
        or multinomial loss.

        For regression tasks with quantile loss, the quantile parameter must be properly
        validated before calling this method."""
        """
        <your code>

class HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting):
    """
    Histogram-based Gradient Boosting Classification Tree.
    
        This estimator is much faster than
        :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
        for big datasets (n_samples >= 10 000).
    
        This estimator has native support for missing values (NaNs). During
        training, the tree grower learns at each split point whether samples
        with missing values should go to the left or right child, based on the
        potential gain. When predicting, samples with missing values are
        assigned to the left or right child consequently. If no missing values
        were encountered for a given feature during training, then samples with
        missing values are mapped to whichever child has the most samples.
    
        This implementation is inspired by
        `LightGBM <https://github.com/Microsoft/LightGBM>`_.
    
        Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.
    
        .. versionadded:: 0.21
    
        Parameters
        ----------
        loss : {'log_loss'}, default='log_loss'
            The loss function to use in the boosting process.
    
            For binary classification problems, 'log_loss' is also known as logistic loss,
            binomial deviance or binary crossentropy. Internally, the model fits one tree
            per boosting iteration and uses the logistic sigmoid function (expit) as
            inverse link function to compute the predicted positive class probability.
    
            For multiclass classification problems, 'log_loss' is also known as multinomial
            deviance or categorical crossentropy. Internally, the model fits one tree per
            boosting iteration and per class and uses the softmax function as inverse link
            function to compute the predicted probabilities of the classes.
    
        learning_rate : float, default=0.1
            The learning rate, also known as *shrinkage*. This is used as a
            multiplicative factor for the leaves values. Use ``1`` for no
            shrinkage.
        max_iter : int, default=100
            The maximum number of iterations of the boosting process, i.e. the
            maximum number of trees for binary classification. For multiclass
            classification, `n_classes` trees per iteration are built.
        max_leaf_nodes : int or None, default=31
            The maximum number of leaves for each tree. Must be strictly greater
            than 1. If None, there is no maximum limit.
        max_depth : int or None, default=None
            The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf.
            Depth isn't constrained by default.
        min_samples_leaf : int, default=20
            The minimum number of samples per leaf. For small datasets with less
            than a few hundred samples, it is recommended to lower this value
            since only very shallow trees would be built.
        l2_regularization : float, default=0
            The L2 regularization parameter penalizing leaves with small hessians.
            Use ``0`` for no regularization (default).
        max_features : float, default=1.0
            Proportion of randomly chosen features in each and every node split.
            This is a form of regularization, smaller values make the trees weaker
            learners and might prevent overfitting.
            If interaction constraints from `interaction_cst` are present, only allowed
            features are taken into account for the subsampling.
    
            .. versionadded:: 1.4
    
        max_bins : int, default=255
            The maximum number of bins to use for non-missing values. Before
            training, each feature of the input array `X` is binned into
            integer-valued bins, which allows for a much faster training stage.
            Features with a small number of unique values may use less than
            ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
            is always reserved for missing values. Must be no larger than 255.
        categorical_features : array-like of {bool, int, str} of shape (n_features)             or shape (n_categorical_features,), default='from_dtype'
            Indicates the categorical features.
    
            - None : no feature will be considered categorical.
            - boolean array-like : boolean mask indicating categorical features.
            - integer array-like : integer indices indicating categorical
              features.
            - str array-like: names of categorical features (assuming the training
              data has feature names).
            - `"from_dtype"`: dataframe columns with dtype "category" are
              considered to be categorical features. The input must be an object
              exposing a ``__dataframe__`` method such as pandas or polars
              DataFrames to use this feature.
    
            For each categorical feature, there must be at most `max_bins` unique
            categories. Negative values for categorical features encoded as numeric
            dtypes are treated as missing values. All categorical values are
            converted to floating point numbers. This means that categorical values
            of 1.0 and 1 are treated as the same category.
    
            Read more in the :ref:`User Guide <categorical_support_gbdt>`.
    
            .. versionadded:: 0.24
    
            .. versionchanged:: 1.2
               Added support for feature names.
    
            .. versionchanged:: 1.4
               Added `"from_dtype"` option.
    
            .. versionchanged:: 1.6
               The default value changed from `None` to `"from_dtype"`.
    
        monotonic_cst : array-like of int of shape (n_features) or dict, default=None
            Monotonic constraint to enforce on each feature are specified using the
            following integer values:
    
            - 1: monotonic increase
            - 0: no constraint
            - -1: monotonic decrease
    
            If a dict with str keys, map feature to monotonic constraints by name.
            If an array, the features are mapped to constraints by position. See
            :ref:`monotonic_cst_features_names` for a usage example.
    
            The constraints are only valid for binary classifications and hold
            over the probability of the positive class.
            Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    
            .. versionadded:: 0.23
    
            .. versionchanged:: 1.2
               Accept dict of constraints with feature names as keys.
    
        interaction_cst : {"pairwise", "no_interactions"} or sequence of lists/tuples/sets             of int, default=None
            Specify interaction constraints, the sets of features which can
            interact with each other in child node splits.
    
            Each item specifies the set of feature indices that are allowed
            to interact with each other. If there are more features than
            specified in these constraints, they are treated as if they were
            specified as an additional set.
    
            The strings "pairwise" and "no_interactions" are shorthands for
            allowing only pairwise or no interactions, respectively.
    
            For instance, with 5 features in total, `interaction_cst=[{0, 1}]`
            is equivalent to `interaction_cst=[{0, 1}, {2, 3, 4}]`,
            and specifies that each branch of a tree will either only split
            on features 0 and 1 or only split on features 2, 3 and 4.
    
            See :ref:`this example<ice-vs-pdp>` on how to use `interaction_cst`.
    
            .. versionadded:: 1.2
    
        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble. For results to be valid, the
            estimator should be re-trained on the same data only.
            See :term:`the Glossary <warm_start>`.
        early_stopping : 'auto' or bool, default='auto'
            If 'auto', early stopping is enabled if the sample size is larger than
            10000 or if `X_val` and `y_val` are passed to `fit`. If True, early stopping
            is enabled, otherwise early stopping is disabled.
    
            .. versionadded:: 0.23
    
        scoring : str or callable or None, default='loss'
            Scoring method to use for early stopping. Only used if `early_stopping`
            is enabled. Options:
    
            - str: see :ref:`scoring_string_names` for options.
            - callable: a scorer callable object (e.g., function) with signature
              ``scorer(estimator, X, y)``. See :ref:`scoring_callable` for details.
            - `None`: :ref:`accuracy <accuracy_score>` is used.
            - 'loss': early stopping is checked w.r.t the loss value.
    
        validation_fraction : int or float or None, default=0.1
            Proportion (or absolute size) of training data to set aside as
            validation data for early stopping. If None, early stopping is done on
            the training data.
            The value is ignored if either early stopping is not performed, e.g.
            `early_stopping=False`, or if `X_val` and `y_val` are passed to fit.
        n_iter_no_change : int, default=10
            Used to determine when to "early stop". The fitting process is
            stopped when none of the last ``n_iter_no_change`` scores are better
            than the ``n_iter_no_change - 1`` -th-to-last one, up to some
            tolerance. Only used if early stopping is performed.
        tol : float, default=1e-7
            The absolute tolerance to use when comparing scores. The higher the
            tolerance, the more likely we are to early stop: higher tolerance
            means that it will be harder for subsequent iterations to be
            considered an improvement upon the reference score.
        verbose : int, default=0
            The verbosity level. If not zero, print some information about the
            fitting process. ``1`` prints only summary info, ``2`` prints info per
            iteration.
        random_state : int, RandomState instance or None, default=None
            Pseudo-random number generator to control the subsampling in the
            binning process, and the train/validation data split if early stopping
            is enabled.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        class_weight : dict or 'balanced', default=None
            Weights associated with classes in the form `{class_label: weight}`.
            If not given, all classes are supposed to have weight one.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as `n_samples / (n_classes * np.bincount(y))`.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if `sample_weight` is specified.
    
            .. versionadded:: 1.2
    
        Attributes
        ----------
        classes_ : array, shape = (n_classes,)
            Class labels.
        do_early_stopping_ : bool
            Indicates whether early stopping is used during training.
        n_iter_ : int
            The number of iterations as selected by early stopping, depending on
            the `early_stopping` parameter. Otherwise it corresponds to max_iter.
        n_trees_per_iteration_ : int
            The number of tree that are built at each iteration. This is equal to 1
            for binary classification, and to ``n_classes`` for multiclass
            classification.
        train_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the training data. The first entry
            is the score of the ensemble before the first iteration. Scores are
            computed according to the ``scoring`` parameter. If ``scoring`` is
            not 'loss', scores are computed on a subset of at most 10 000
            samples. Empty if no early stopping.
        validation_score_ : ndarray, shape (n_iter_+1,)
            The scores at each iteration on the held-out validation data. The
            first entry is the score of the ensemble before the first iteration.
            Scores are computed according to the ``scoring`` parameter. Empty if
            no early stopping or if ``validation_fraction`` is None.
        is_categorical_ : ndarray, shape (n_features, ) or None
            Boolean mask for the categorical features. ``None`` if there are no
            categorical features.
        n_features_in_ : int
            Number of features seen during :term:`fit`.
    
            .. versionadded:: 0.24
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
    
            .. versionadded:: 1.0
    
        See Also
        --------
        GradientBoostingClassifier : Exact gradient boosting method that does not
            scale as good on datasets with a large number of samples.
        sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
        RandomForestClassifier : A meta-estimator that fits a number of decision
            tree classifiers on various sub-samples of the dataset and uses
            averaging to improve the predictive accuracy and control over-fitting.
        AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
            on the original dataset and then fits additional copies of the
            classifier on the same dataset where the weights of incorrectly
            classified instances are adjusted such that subsequent classifiers
            focus more on difficult cases.
    
        Examples
        --------
        >>> from sklearn.ensemble import HistGradientBoostingClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = HistGradientBoostingClassifier().fit(X, y)
        >>> clf.score(X, y)
        1.0
        
    """

    _parameter_constraints = "{**BaseHistGradientBoosting._parameter_constraints, 'loss': [StrOptions({'log_loss'}), BaseLoss], 'class_weight': [dict, StrOptions({'balanced'}), None]}"

    def __init__(self, loss = 'log_loss'):
        """
        Initialize a histogram-based gradient boosting classifier.

        This constructor sets up the basic parameters for the HistGradientBoostingClassifier,
        which implements a fast gradient boosting algorithm using histogram-based tree growing.

        Parameters
        ----------
        loss : {'log_loss'} or BaseLoss instance, default='log_loss'
            The loss function to use in the boosting process. For binary classification,
            'log_loss' uses logistic loss (binomial deviance). For multiclass classification,
            'log_loss' uses multinomial deviance (categorical crossentropy). Custom loss
            functions can be provided as BaseLoss instances.

        learning_rate : float, default=0.1
            The learning rate (shrinkage factor) applied to each tree's contribution.
            Lower values make the model more robust but require more iterations.
            Must be greater than 0.

        max_iter : int, default=100
            Maximum number of boosting iterations. For binary classification, this is
            the number of trees. For multiclass, n_classes trees are built per iteration.
            Must be at least 1.

        max_leaf_nodes : int or None, default=31
            Maximum number of leaves per tree. Must be greater than 1 if specified.
            If None, no limit is imposed on the number of leaves.

        max_depth : int or None, default=None
            Maximum depth of each tree (number of edges from root to deepest leaf).
            If None, depth is not constrained.

        min_samples_leaf : int, default=20
            Minimum number of samples required in each leaf node. For small datasets,
            consider lowering this value to allow deeper trees.

        l2_regularization : float, default=0.0
            L2 regularization parameter for leaf values. Higher values provide more
            regularization. Must be non-negative.

        max_features : float, default=1.0
            Proportion of features to consider at each split (0 < max_features <= 1).
            Provides regularization by introducing randomness in feature selection.

        max_bins : int, default=255
            Maximum number of bins for discretizing continuous features. Must be
            between 2 and 255. One additional bin is reserved for missing values.

        categorical_features : array-like, str or None, default='from_dtype'
            Specification of categorical features. Can be boolean mask, integer indices,
            feature names, or 'from_dtype' to auto-detect from pandas categorical columns.

        monotonic_cst : array-like, dict or None, default=None
            Monotonic constraints for features. Values: 1 (increasing), 0 (no constraint),
            -1 (decreasing). Only supported for binary classification.

        interaction_cst : sequence, str or None, default=None
            Interaction constraints specifying which features can interact in splits.
            Can be 'pairwise', 'no_interactions', or custom feature groupings.

        warm_start : bool, default=False
            Whether to reuse the previous fit solution and add more estimators.
            Requires fitting on the same data.

        early_stopping : 'auto' or bool, default='auto'
            Whether to use early stopping. 'auto' enables it for large datasets
            (n_samples > 10000) or when validation data is provided.

        scoring : str, callable or None, default='loss'
            Scoring method for early stopping. Can be a scorer name, callable,
            None (uses accuracy), or 'loss' (uses the loss function).

        validation_fraction : float, int or None, default=0.1
            Fraction or absolute size of training data for validation in early stopping.
            Ignored if validation data is explicitly provided or early stopping is disabled.

        n_iter_no_change : int, default=10
            Number of iterations with no improvement to trigger early stopping.
            Must be at least 1.

        tol : float, default=1e-7
            Tolerance for early stopping. Higher values make early stopping more likely.
            Must be non-negative.

        verbose : int, default=0
            Verbosity level. 0 for silent, 1 for summary info, 2 for detailed iteration info.

        random_state : int, RandomState or None, default=None
            Random state for reproducible results. Controls binning randomness and
            train/validation splits.

        class_weight : dict, 'balanced' or None, default=None
            Weights for classes. 'balanced' uses inverse class frequencies.
            Custom weights as {class_label: weight} dictionary.

        Notes
        -----
        This initializer only sets the parameters. The actual model training occurs
        when the fit method is called. The classifier supports both binary and
        multiclass classification automatically based on the target variable.

        For binary classification, one tree per iteration is built. For multiclass
        classification, one tree per class per iteration is built.
        """
        <your code>

    def _finalize_sample_weight(self, sample_weight, y):
        """
        Adjust sample_weights with class_weights.
        """
        <your code>

    def predict(self, X):
        """
        Predict classes for X.

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.

                Returns
                -------
                y : ndarray, shape (n_samples,)
                    The predicted classes.

        """
        <your code>

    def staged_predict(self, X):
        """
        Predict classes at each iteration.

                This method allows monitoring (i.e. determine error on testing set)
                after each stage.

                .. versionadded:: 0.24

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The input samples.

                Yields
                ------
                y : generator of ndarray of shape (n_samples,)
                    The predicted classes of the input samples, for each iteration.

        """
        <your code>

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.

                Returns
                -------
                p : ndarray, shape (n_samples, n_classes)
                    The class probabilities of the input samples.

        """
        <your code>

    def staged_predict_proba(self, X):
        """
        Predict class probabilities at each iteration.

                This method allows monitoring (i.e. determine error on testing set)
                after each stage.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The input samples.

                Yields
                ------
                y : generator of ndarray of shape (n_samples,)
                    The predicted class probabilities of the input samples,
                    for each iteration.

        """
        <your code>

    def decision_function(self, X):
        """
        Compute the decision function of ``X``.

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.

                Returns
                -------
                decision : ndarray, shape (n_samples,) or                 (n_samples, n_trees_per_iteration)
                    The raw predicted values (i.e. the sum of the trees leaves) for
                    each sample. n_trees_per_iteration is equal to the number of
                    classes in multiclass classification.

        """
        <your code>

    def staged_decision_function(self, X):
        """
        Compute decision function of ``X`` for each iteration.

                This method allows monitoring (i.e. determine error on testing set)
                after each stage.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    The input samples.

                Yields
                ------
                decision : generator of ndarray of shape (n_samples,) or                 (n_samples, n_trees_per_iteration)
                    The decision function of the input samples, which corresponds to
                    the raw values predicted from the trees of the ensemble . The
                    classes corresponds to that in the attribute :term:`classes_`.

        """
        <your code>

    def _encode_y(self, y):
        """
        Create self._label_encoder and encode y correspondingly.
        """
        <your code>

    def _encode_y_val(self, y):
        """
        Encode validation target values for internal use during fitting.

        This method transforms validation target values (y_val) to the same encoded format
        used internally by the gradient boosting algorithm. For regressors, this typically
        involves dtype conversion and validation. For classifiers, this involves label
        encoding to convert class labels to integer indices.

        Parameters
        ----------
        y : array-like of shape (n_samples,), default=None
            The validation target values to encode. These should be in the same format
            as the original training targets passed to fit().

        Returns
        -------
        encoded_y : ndarray of shape (n_samples,)
            The encoded validation target values. For regressors, this is typically
            the input array converted to the appropriate dtype (Y_DTYPE). For classifiers,
            this is the label-encoded version where class labels are converted to
            integer indices corresponding to the classes learned during training.

        Notes
        -----
        This method is called internally during the fit process when validation data
        is provided (either through validation_fraction parameter or explicit X_val/y_val
        parameters). It ensures that validation targets are in the same format as the
        training targets for consistent evaluation during early stopping.

        For classifiers, this method uses the label encoder fitted on the training data
        to transform validation labels, ensuring consistency between training and
        validation label encoding.

        For regressors, this method applies the same validation and preprocessing
        steps as _encode_y to ensure validation targets meet the same requirements
        as training targets (e.g., positivity constraints for gamma loss).
        """
        <your code>

    def _get_loss(self, sample_weight):
        """
        """Get the appropriate loss function for the gradient boosting model.

        This method creates and returns a loss function instance based on the model's
        loss parameter and current configuration. The loss function is used throughout
        the boosting process to compute gradients and hessians for tree fitting.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,) or None
            Sample weights for the training data. If None, uniform weights are assumed.
            The loss function uses these weights to properly weight the contribution of
            each sample during gradient and hessian computation.

        Returns
        -------
        loss : BaseLoss
            An instance of the appropriate loss function class. The specific type depends
            on the model's loss parameter:

            - For regression: Returns instances like HalfSquaredError, HalfPoissonLoss,
              HalfGammaLoss, or PinballLoss based on the loss string parameter
            - For classification: Returns HalfBinomialLoss for binary classification
              or HalfMultinomialLoss for multiclass classification

            The returned loss instance is configured with the provided sample_weight
            and any additional parameters specific to the loss type (e.g., quantile
            parameter for PinballLoss).

        Notes
        -----
        This is an abstract method that must be implemented by subclasses. The
        implementation varies between HistGradientBoostingRegressor and
        HistGradientBoostingClassifier to handle their respective loss functions
        and requirements.

        The loss function instance is used internally during the boosting process
        to compute gradients and hessians, evaluate training/validation scores,
        and make predictions through the inverse link function.

        For custom loss functions passed as BaseLoss instances rather than strings,
        this method should return the loss instance directly after any necessary
        configuration."""
        """
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.

### Interface Description 2
Below is **Interface Description 2** for file: sklearn-ensemble-_hist_gradient_boosting-grower.py

This file contains 1 top-level interface(s) that need to be implemented.

```python
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
        """
        Initialize a TreeGrower instance for building gradient boosting regression trees.

        TreeGrower constructs regression trees by fitting Newton-Raphson steps based on
        gradients and hessians of training samples. It uses a best-first splitting strategy
        where split quality is determined by gain calculations in the splitting module.

        Parameters
        ----------
        X_binned : ndarray of shape (n_samples, n_features), dtype=np.uint8
            The binned input training samples. Must be Fortran-aligned for optimal
            performance. Each feature's values should be discretized into bins.
        gradients : ndarray of shape (n_samples,)
            The gradients of the loss function with respect to predictions, evaluated
            at the previous boosting iteration (i-1). Used to determine split directions.
        hessians : ndarray of shape (n_samples,)
            The second derivatives (hessians) of the loss function with respect to
            predictions, evaluated at the previous boosting iteration (i-1). Used for
            Newton-Raphson step calculations.
        max_leaf_nodes : int, default=None
            Maximum number of leaf nodes allowed in the tree. If None, no limit is
            imposed on the number of leaves.
        max_depth : int, default=None
            Maximum depth of the tree, measured as the number of edges from root to
            the deepest leaf. If None, depth is not constrained.
        min_samples_leaf : int, default=20
            Minimum number of samples required to form a leaf node. Splits resulting
            in leaves with fewer samples are rejected.
        min_gain_to_split : float, default=0.0
            Minimum improvement in loss reduction required to make a split. Splits
            with gain below this threshold are discarded.
        min_hessian_to_split : float, default=0.001
            Minimum sum of hessians required in a node to consider splitting it.
            Helps prevent splits on nodes with very low confidence.
        n_bins : int, default=256
            Total number of bins used for feature discretization, including the
            special bin reserved for missing values.
        n_bins_non_missing : ndarray of shape (n_features,) or int, default=None
            Number of bins actually used for non-missing values per feature. If int,
            same value applies to all features. If None, defaults to n_bins - 1 for
            all features.
        has_missing_values : bool or ndarray of shape (n_features,), default=False
            Indicates whether each feature contains missing values in training data.
            If bool, same value applies to all features.
        is_categorical : ndarray of shape (n_features,), dtype=bool, default=None
            Boolean array indicating which features are categorical. If None, all
            features are treated as numerical.
        monotonic_cst : array-like of shape (n_features,), dtype=int, default=None
            Monotonic constraints for each feature. Values: 1 (increasing), 0 (no
            constraint), -1 (decreasing). If None, no constraints are applied.
        interaction_cst : list of sets of int, default=None
            Interaction constraints specifying which features can be used together
            in the same path from root to leaf. Each set contains feature indices
            that can interact.
        l2_regularization : float, default=0.0
            L2 regularization parameter applied to leaf values. Higher values
            penalize leaves with small hessians, helping prevent overfitting.
        feature_fraction_per_split : float, default=1.0
            Fraction of features to randomly consider at each split. Values less
            than 1.0 provide regularization by making trees weaker learners.
        rng : numpy.random.Generator, default=np.random.default_rng()
            Random number generator used for feature subsampling when
            feature_fraction_per_split < 1.0.
        shrinkage : float, default=1.0
            Learning rate applied to leaf values. Also known as the shrinkage
            parameter in gradient boosting literature.
        n_threads : int, default=None
            Number of OpenMP threads for parallel computation. If None, determined
            automatically considering system resources and cgroup constraints.

        Raises
        ------
        NotImplementedError
            If X_binned is not of dtype uint8.
        ValueError
            If X_binned is not Fortran-contiguous, min_gain_to_split is negative,
            min_hessian_to_split is negative, or categorical features have monotonic
            constraints.

        Notes
        -----
        The tree growing process uses histogram-based splitting for efficiency. The
        algorithm maintains a priority queue of splittable nodes, always splitting
        the node with the highest potential gain first. Categorical features are
        handled using bitset representations for efficient subset splitting.

        When monotonic constraints are specified, child node value bounds are
        enforced to ensure the learned function respects the constraints. Interaction
        constraints limit which features can appear together in any root-to-leaf path.
        """
        <your code>

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
        <your code>

    def grow(self):
        """
        Grow the tree, from root to leaves.
        """
        <your code>

    def _apply_shrinkage(self):
        """
        Multiply leaves values by shrinkage parameter.

                This must be done at the very end of the growing process. If this were
                done during the growing process e.g. in finalize_leaf(), then a leaf
                would be shrunk but its sibling would potentially not be (if it's a
                non-leaf), which would lead to a wrong computation of the 'middle'
                value needed to enforce the monotonic constraints.

        """
        <your code>

    def _initialize_root(self):
        """
        Initialize root node and finalize it if needed.
        """
        <your code>

    def _compute_best_split_and_push(self, node):
        """
        Compute the best possible split (SplitInfo) of a given node.

                Also push it in the heap of splittable nodes if gain isn't zero.
                The gain of a node is 0 if either all the leaves are pure
                (best gain = 0), or if no split would satisfy the constraints,
                (min_hessians_to_split, min_gain_to_split, min_samples_leaf)

        """
        <your code>

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
        <your code>

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
        <your code>

    def _finalize_leaf(self, node):
        """
        Make node a leaf of the tree being grown.
        """
        <your code>

    def _finalize_splittable_nodes(self):
        """
        Transform all splittable nodes into leaves.

                Used when some constraint is met e.g. maximum number of leaves or
                maximum depth.
        """
        <your code>

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
        <your code>

```

Remember, **the interface template above is extremely important**. You must generate callable interfaces strictly according to the specified requirements, as this will directly determine whether you can pass our tests. If your implementation has incorrect naming or improper input/output formats, it may directly result in a 0% pass rate for this case.