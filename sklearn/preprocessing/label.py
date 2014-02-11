# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Rohit Sivaprasad
#
# License: BSD 3 clause

import warnings
import array
import numpy as np

from scipy.sparse import csr_matrix, issparse

from ..base import BaseEstimator, TransformerMixin

from ..utils.fixes import unique, np_version
from ..utils import deprecated, column_or_1d

from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target

from ..externals import six

zip = six.moves.zip
map = six.moves.map

__all__ = [
    'label_binarize',
    'LabelBinarizer',
    'LabelEncoder',
]


def _check_numpy_unicode_bug(labels):
    """Check that user is not subject to an old numpy bug

    Fixed in master before 1.7.0:

      https://github.com/numpy/numpy/pull/243

    """
    if np_version[:3] < (1, 7, 0) and labels.dtype.kind == 'U':
        raise RuntimeError("NumPy < 1.7.0 does not implement searchsorted"
                           " on unicode data correctly. Please upgrade"
                           " NumPy to use LabelEncoder with unicode inputs.")


class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.

    Attributes
    ----------
    `classes_` : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    """

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelEncoder was not fitted yet.")

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        _check_numpy_unicode_bug(y)
        self.classes_ = np.unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        y = column_or_1d(y, warn=True)
        _check_numpy_unicode_bug(y)
        self.classes_, y = unique(y, return_inverse=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self._check_fitted()

        classes = np.unique(y)
        _check_numpy_unicode_bug(classes)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        self._check_fitted()

        y = np.asarray(y)
        return self.classes_[y]


class LabelBinarizer(BaseEstimator, TransformerMixin):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Parameters
    ----------

    neg_label : int (default: 0)
        Value with which negative labels must be encoded.

    pos_label : int (default: 1)
        Value with which positive labels must be encoded.

    dense_output : boolean,
        If True, the transformer will yield a csr sparse matrix, except
        with binary target type.

    Attributes
    ----------
    `classes_` : array of shape [n_class]
        Holds the label for each class.

    `multilabel_` : boolean
        True if the transformer was fitted on a multilabel rather than a
        multiclass set of labels.

    `y_type_` : str,
        Give the type target detected at fitting time, e.g. "binary",
        "multiclass", "multilabel-indicator", "multilabel-sequences".

    Examples
    --------
    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(dense_output=True, neg_label=0, pos_label=1)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.multilabel_
    False
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])
    >>> lb.multilabel_
    True

    See also
    --------
    label_binarize : function to perform the transform operation of
        LabelBinarizer with fixed classes.
    """

    def __init__(self, neg_label=0, pos_label=1, dense_output=True):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.dense_output = dense_output

    @property
    @deprecated("Attribute `multilabel` was renamed to `multilabel_` in "
                "0.14 and will be removed in 0.16")
    def multilabel(self):
        return self.multilabel_

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelBinarizer was not fitted yet.")

    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : numpy array of shape (n_samples,) or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.neg_label >= self.pos_label:
            raise ValueError("neg_label={0} must be strictly less than "
                             "pos_label={1}.".format(self.neg_label,
                                                     self.pos_label))

        self.y_type_ = type_of_target(y)

        self.multilabel_ = self.y_type_.startswith('multilabel')
        if self.multilabel_:
            self.indicator_matrix_ = self.y_type_ == 'multilabel-indicator'


        if self.y_type_ == "multilabel-indicator":
            self.sparse_indicator_matrix_ = issparse(y)

        self.classes_ = unique_labels(y)

        return self

    def transform(self, y):
        """Transform multi-class labels to binary labels

        The output of transform is sometimes referred to by some authors as the
        1-of-K coding scheme.

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes]
        """
        self._check_fitted()
        return label_binarize(y, self.classes_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label,
                              dense_output=self.dense_output)


    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array of shape [n_samples, n_classes]
            Target values.

        threshold : float or None
            Threshold used in the binary and multi-label cases.

            Use 0 when:
                - Y contains the output of decision_function (classifier)
            Use 0.5 when:
                - Y contains the output of predict_proba

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        self._check_fitted()

        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.

        y_inv = _inverse_label_binarize(Y, self.y_type_, self.classes_,
                                        threshold)

        if self.y_type_ == "multilabel-indicator" :
            if self.sparse_indicator_matrix_:
                y_inv = csr_matrix(y_inv)
            elif issparse(y_inv):
                y_inv = y_inv.toarray()

        return y_inv


def label_binarize(y, classes, neg_label=0, pos_label=1,
                   dense_output=True, multilabel=None):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape [n_classes]
        Uniquely holds the label for each class.

    neg_label: int (default: 0)
        Value with which negative labels must be encoded.

    pos_label: int (default: 1)
        Value with which positive labels must be encoded.

    dense_output : boolean, optional (default=True)
        If True, ensure that the output of label_binarize is a
        dense numpy array even if the binarize matrix is sparse.
        If False, the binarized data uses a sparse representation.
        With "binary" type, ``dense_output`` is set to false.


    Returns
    -------
    Y : numpy array or csr matrix with shape [n_samples, n_classes]

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    >>> label_binarize([(1, 2), (6,), ()], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 1],
           [0, 1, 0, 0],
           [0, 0, 0, 0]])

    See also
    --------
    LabelBinarizer

    """
    if neg_label >= pos_label:
        raise ValueError("neg_label must be strictly less than pos_label.")

    if ((pos_label == 0 or neg_label != 0) and dense_output == False):
        raise ValueError("Sparse binarization is only supported with non zero "
                         "pos_label")

    # To account for pos_label==0 in the dense case
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = - neg_label


    y_type = type_of_target(y)

    if y_type == "binary" and len(classes) == 1:
        classes = np.insert(classes, [0], 0)

    n_samples = y.shape[0] if issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)
    sorted_class = np.sort(classes)
    if (y_type == "multilabel-indicator" and classes.size != y.shape[1] or
            not set(classes).issuperset(unique_labels(y))):
        raise ValueError("classes {0} missmatch with the data".format(classes))


    if y_type == "binary" and len(classes) > 2:
        y_type = "multiclass"

    if y_type == "binary":
        dense_output = True


    if multilabel is not None:
        warnings.warn("The multilabel parameter is deprecated as of version "
                      "0.15 and will be removed in 0.17.", DeprecationWarning)

    if not dense_output and neg_label != 0:
        raise ValueError("Non-zero neg_label is not supported with "
                         "dense_output=False if y is not binary")

    if y_type in ("binary", "multiclass"):
        y = column_or_1d(y)
        indptr = np.arange(n_samples + 1)
        indices = np.searchsorted(sorted_class, y)
        data = np.empty_like(indices)
        data.fill(pos_label)

        Y = csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))

    elif y_type == "multilabel-indicator":
        Y = csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data

    elif y_type == "multilabel-sequences":
        indptr = array.array('i', [0])
        indices = array.array('i')
        for i, label_sequence in enumerate(y):
            y_i = np.searchsorted(sorted_class, np.unique(label_sequence))
            indices.extend(y_i)
            indptr.append(indptr[-1] + len(y_i))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
    else:
        raise ValueError("{0} format is not supported".format(y_type))

    if dense_output:
        Y = Y.toarray()

        if neg_label != 0:
            Y[Y == 0] = neg_label

        if pos_switch:
            Y[Y == pos_label] = 0

    # preserve label ordering
    if np.any(classes != sorted_class):
        indices = np.argsort(classes)
        Y = Y[:, indices]

    if y_type == "binary":
        Y = Y[:, 1].reshape((-1, 1))

    return Y


def _inverse_label_binarize(y, y_type, classes, threshold):
    """Inverse label binarization transformation"""

    if y_type == "binary" and  y.ndim == 2 and y.shape[1] != 1:
         raise ValueError("y_type='binary', but y.shape = {0}".format(y.shape))

    if y_type != "binary" and y.shape[1] != len(classes):
        raise ValueError("The number of class is not equal to the number of "
                         "dimension of y.")

    classes = np.asarray(classes)

    # Perform thresholding
    if issparse(y):
        if threshold > 0:
            y = y.tocsr()
            y.data = np.array(y.data > threshold, dtype=np.int)
            y.eliminate_zeros()
        else:
            y = np.array(y.toarray() > threshold, dtype=np.int)

    else:
        y = np.array(y > threshold, dtype=np.int)

    # Inverse transform data
    if y_type == "binary":
        if issparse(y):
            y = y.toarray()

        return classes[y.ravel()]

    elif y_type == "multiclass":
        if issparse(y):
            return np.array([classes[y.indices[start +
                                               y.data[start:end].argmax()]]
                             for start, end in zip(y.indptr[:-1],
                                                   y.indptr[1:])])
        else:
            return classes[y.argmax(axis=1)]

    elif y_type == "multilabel-indicator":
        return y

    elif y_type == "multilabel-sequences":
        if issparse(y):
            return [tuple(classes.take(y.indices[start:end]))
                    for start, end in zip(y.indptr[:-1], y.indptr[1:])]
        else:
            return [tuple(classes.take(np.flatnonzero(y[i])))
                          for i in range(len(y))]

    else:
        raise ValueError("{0} format is not supported".format(y_type))
