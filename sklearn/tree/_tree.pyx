# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#
# Licence: BSD 3 clause

from libc.stdlib cimport calloc, free, malloc, realloc
from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from cpython cimport Py_INCREF, PyObject

from sklearn.tree._utils cimport Stack, StackRecord
from sklearn.tree._utils cimport PriorityHeap, PriorityHeapRecord

import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides, void* data, int flags,
                                object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef double EPSILON_DBL = 1e-7
cdef float EPSILON_FLT = 1e-7
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_samples,
    ]
})


# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    """Interface for impurity criteria."""

    cdef void init(self,
                   DOUBLE_t* y,
                   SIZE_t y_stride,
                   DOUBLE_t* sample_weight,
                   SIZE_t* samples,
                   SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        pass

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        pass

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        pass

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        pass

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of
           samples[start:pos] + the impurity of samples[pos:end]."""
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        pass

    cdef double impurity_improvement(self, double impurity) nogil:
        """Impurity improvement, i.e.
           impurity - (left impurity + right impurity)."""
        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return (impurity -
                (self.weighted_n_right /
                 self.weighted_n_node_samples * impurity_right) -
                (self.weighted_n_left /
                 self.weighted_n_node_samples * impurity_left))


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""
    cdef SIZE_t* n_classes
    cdef SIZE_t label_count_stride
    cdef double* label_count_left
    cdef double* label_count_right
    cdef double* label_count_total

    def __cinit__(self, SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes):
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.label_count_left = NULL
        self.label_count_right = NULL
        self.label_count_total = NULL

        # Count labels for each output
        self.n_classes = <SIZE_t*> malloc(n_outputs * sizeof(SIZE_t))
        if self.n_classes == NULL:
            raise MemoryError()

        cdef SIZE_t k = 0
        cdef SIZE_t label_count_stride = 0

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > label_count_stride:
                label_count_stride = n_classes[k]

        self.label_count_stride = label_count_stride

        # Allocate counters
        cdef SIZE_t n_bytes = n_outputs * label_count_stride
        self.label_count_left = <double*> calloc(n_bytes, sizeof(double))
        self.label_count_right = <double*> calloc(n_bytes, sizeof(double))
        self.label_count_total = <double*> calloc(n_bytes, sizeof(double))

        # Check for allocation errors
        if (self.label_count_left == NULL or
                self.label_count_right == NULL or
                self.label_count_total == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)
        free(self.label_count_left)
        free(self.label_count_right)
        free(self.label_count_total)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        cdef double weighted_n_node_samples = 0.0

        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(n_outputs):
            memset(label_count_total + offset, 0, n_classes[k] * sizeof(double))
            offset += label_count_stride

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                label_count_total[k * label_count_stride + c] += w

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            memset(label_count_left, 0, n_classes[k] * sizeof(double))
            memcpy(label_count_right, label_count_total,
                   n_classes[k] * sizeof(double))

            label_count_total += label_count_stride
            label_count_left += label_count_stride
            label_count_right += label_count_stride

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0

        # Note: We assume start <= pos < new_pos <= end

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                label_index = (k * label_count_stride +
                               <SIZE_t> y[i * y_stride + k])
                label_count_left[label_index] += w
                label_count_right[label_index] -= w

            weighted_n_left += w
            weighted_n_right -= w

        self.weighted_n_left = weighted_n_left
        self.weighted_n_right = weighted_n_right

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef SIZE_t k

        for k in range(n_outputs):
            memcpy(dest, label_count_total, n_classes[k] * sizeof(double))
            dest += label_count_stride
            label_count_total += label_count_stride


cdef class Entropy(ClassificationCriterion):
    """Cross Entropy impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double entropy = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_total[c]
                if tmp > 0.0:
                    tmp /= weighted_n_node_samples
                    entropy -= tmp * log(tmp)

            total += entropy
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy_left = 0.0
            entropy_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]
                if tmp > 0.0:
                    tmp /= weighted_n_left
                    entropy_left -= tmp * log(tmp)

                tmp = label_count_right[c]
                if tmp > 0.0:
                    tmp /= weighted_n_right
                    entropy_right -= tmp * log(tmp)

            total_left += entropy_left
            total_right += entropy_right
            label_count_left += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs


cdef class Gini(ClassificationCriterion):
    """Gini Index impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} pmk (1 - pmk)
              = 1 - \sum_{k=0}^{K-1} pmk ** 2
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double gini = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_total[c]  # TODO: use weighted count instead
                gini += tmp * tmp

            gini = 1.0 - gini / (weighted_n_node_samples *
                                 weighted_n_node_samples)

            total += gini
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double total = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini_left = 0.0
            gini_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]   # TODO: use weighted count instead
                gini_left += tmp * tmp
                tmp = label_count_right[c]
                gini_right += tmp * tmp

            gini_left = 1.0 - gini_left / (weighted_n_left *
                                           weighted_n_left)
            gini_right = 1.0 - gini_right / (weighted_n_right *
                                             weighted_n_right)

            total_left += gini_left
            total_right += gini_right
            label_count_left += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs


cdef class RegressionCriterion(Criterion):
    """Abstract criterion for regression.

    Computes variance of the target values left and right of the split point.
    Computation is linear in `n_samples` by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples y_bar ** 2
    """
    cdef double* mean_left
    cdef double* mean_right
    cdef double* mean_total
    cdef double* sq_sum_left
    cdef double* sq_sum_right
    cdef double* sq_sum_total
    cdef double* var_left
    cdef double* var_right
    cdef double* sum_left
    cdef double* sum_right
    cdef double* sum_total

    def __cinit__(self, SIZE_t n_outputs):
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.mean_left = NULL
        self.mean_right = NULL
        self.mean_total = NULL
        self.sq_sum_left = NULL
        self.sq_sum_right = NULL
        self.sq_sum_total = NULL
        self.var_left = NULL
        self.var_right = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.sum_total = NULL

        self.mean_left = <double*> calloc(n_outputs, sizeof(double))
        self.mean_right = <double*> calloc(n_outputs, sizeof(double))
        self.mean_total = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.var_left = <double*> calloc(n_outputs, sizeof(double))
        self.var_right = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))

        if (self.mean_left == NULL or
                self.mean_right == NULL or
                self.mean_total == NULL or
                self.sq_sum_left == NULL or
                self.sq_sum_right == NULL or
                self.sq_sum_total == NULL or
                self.var_left == NULL or
                self.var_right == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL or
                self.sum_total == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.mean_left)
        free(self.mean_right)
        free(self.mean_total)
        free(self.sq_sum_left)
        free(self.sq_sum_right)
        free(self.sq_sum_total)
        free(self.var_left)
        free(self.var_right)
        free(self.sum_left)
        free(self.sum_right)
        free(self.sum_total)

    def __reduce__(self):
        return (RegressionCriterion, (self.n_outputs,), self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        cdef double weighted_n_node_samples = 0.

        # Initialize accumulators
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef DOUBLE_t y_ik = 0.0
        cdef DOUBLE_t w_y_ik = 0.0
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t n_bytes = n_outputs * sizeof(double)
        memset(mean_left, 0, n_bytes)
        memset(mean_right, 0, n_bytes)
        memset(mean_total, 0, n_bytes)
        memset(sq_sum_left, 0, n_bytes)
        memset(sq_sum_right, 0, n_bytes)
        memset(sq_sum_total, 0, n_bytes)
        memset(var_left, 0, n_bytes)
        memset(var_right, 0, n_bytes)
        memset(sum_left, 0, n_bytes)
        memset(sum_right, 0, n_bytes)
        memset(sum_total, 0, n_bytes)

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                sum_total[k] += w_y_ik
                sq_sum_total[k] += w_y_ik * y_ik

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

        for k in range(n_outputs):
            mean_total[k] = sum_total[k] / weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            mean_right[k] = mean_total[k]
            mean_left[k] = 0.0
            sq_sum_right[k] = sq_sum_total[k]
            sq_sum_left[k] = 0.0
            var_right[k] = (sq_sum_right[k] / weighted_n_node_samples -
                            mean_right[k] * mean_right[k])
            var_left[k] = 0.0
            sum_right[k] = sum_total[k]
            sum_left[k] = 0.0

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_ik, w_y_ik

        # Note: We assume start <= pos < new_pos <= end
        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik

                sum_left[k] += w_y_ik
                sum_right[k] -= w_y_ik

                sq_sum_left[k] += w_y_ik * y_ik
                sq_sum_right[k] -= w_y_ik * y_ik

            weighted_n_left += w
            weighted_n_right -= w

        for k in range(n_outputs):
            mean_left[k] = sum_left[k] / weighted_n_left
            mean_right[k] = sum_right[k] / weighted_n_right
            var_left[k] = (sq_sum_left[k] / weighted_n_left -
                           mean_left[k] * mean_left[k])
            var_right[k] = (sq_sum_right[k] / weighted_n_right -
                            mean_right[k] * mean_right[k])

        self.weighted_n_left = weighted_n_left
        self.weighted_n_right = weighted_n_right

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        memcpy(dest, self.mean_total, self.n_outputs * sizeof(double))


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* mean_total = self.mean_total
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double total = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total += (sq_sum_total[k] / weighted_n_node_samples -
                      mean_total[k] * mean_total[k])

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total_left += var_left[k]
            total_right += var_right[k]

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman.

    Uses the formula (35) in Friedmans original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t k
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right
        cdef double diff = 0.0

        for k from 0 <= k < n_outputs:
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        total_sum_left = total_sum_left / n_outputs
        total_sum_right = total_sum_right / n_outputs
        diff = ((total_sum_left / weighted_n_left) -
                (total_sum_right / weighted_n_right))

        return (weighted_n_left * weighted_n_right * diff * diff) / (weighted_n_left + weighted_n_right)


# =============================================================================
# Splitter
# =============================================================================

cdef class Splitter:
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, object random_state):
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.X = NULL
        self.X_sample_stride = 0
        self.X_fx_stride = 0
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""
        free(self.samples)
        free(self.features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight):
        """Initialize the splitter."""
        # Reset random state
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)

        # Initialize samples and features structures
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t* samples = <SIZE_t*> realloc(self.samples,
                                                 n_samples * sizeof(SIZE_t))
        if samples == NULL:
            raise MemoryError()

        cdef SIZE_t i, j
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

        self.samples = samples
        self.n_samples = j

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = <SIZE_t*> realloc(self.features,
                                                  n_features * sizeof(SIZE_t))
        if features == NULL:
            raise MemoryError()

        for i in range(n_features):
            features[i] = i

        self.features = features
        self.n_features = n_features

        cdef DTYPE_t* fv = <DTYPE_t*> realloc(self.feature_values,
                                              n_samples * sizeof(DTYPE_t))
        if fv == NULL:
            raise MemoryError()
        self.feature_values = fv

        # Initialize X, y, sample_weight
        self.X = <DTYPE_t*> X.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        self.sample_weight = sample_weight

    cdef void node_reset(self, SIZE_t start, SIZE_t end) nogil:
        """Reset splitter on node samples[start:end]."""
        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.samples,
                            start,
                            end)

    cdef void node_split(self, double impurity, SIZE_t* pos, SIZE_t* feature,
                         double* threshold, double* impurity_left,
                         double* impurity_right, double* impurity_improvement,
                         SIZE_t* n_valid_features) nogil:
        """Find a split on node samples[start:end]."""
        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SIZE_t* pos, SIZE_t* feature,
                         double* threshold, double* impurity_left,
                         double* impurity_right, double* impurity_improvement,
                         SIZE_t* n_valid_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        # To avoid trying to split on constant features:
        # the array `features` is such that
        # `features[:n_invalid]` corresponds to features with
        # constant values at the current depth,
        # `features[:n_invalid_drawn]` corresponds to features that are
        # constant and were randomly drawn by the shuffling algorithm.
        cdef SIZE_t* features = self.features
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_invalid = n_features - n_valid_features[0]
        cdef SIZE_t n_invalid_drawn = 0

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef double best_impurity_left = INFINITY
        cdef double best_impurity_right = INFINITY
        cdef SIZE_t best_pos = end
        cdef SIZE_t best_feature = 0
        cdef double best_threshold = 0.
        cdef double best_improvement = -INFINITY

        cdef double current_improvement
        cdef double current_impurity
        cdef double current_impurity_left
        cdef double current_impurity_right
        cdef SIZE_t current_pos
        cdef SIZE_t current_feature
        cdef double current_threshold

        cdef SIZE_t f_idx, f_i, f_j, p, tmp
        cdef SIZE_t visited_features = 0

        cdef SIZE_t partition_start
        cdef SIZE_t partition_end

        for f_idx in range(n_features):
            # Draw a feature at random
            f_i = n_invalid_drawn + n_features - f_idx - 1
            f_j = n_invalid_drawn + rand_int(n_features - f_idx, random_state)

            if f_j < n_invalid:
                # Swap f_j and n_invalid_drawn
                tmp = features[f_j]
                features[f_j] = features[n_invalid_drawn]
                features[n_invalid_drawn] = tmp
                n_invalid_drawn += 1
                visited_features += 1
                continue

            features[f_i], features[f_j] = features[f_j], features[f_i]
            current_feature = features[f_i]

            # Sort samples along that feature; first copy the feature values
            # for the active samples into Xf, s.t. Xf[i] == X[samples[i], j],
            # so the sort uses the cache more effectively.
            for p in range(start, end):
                Xf[p] = X[X_sample_stride * samples[p] +
                          X_fx_stride * current_feature]

            sort(Xf + start, samples + start, end - start)

            # The feature is constant since Xf[start] ~ Xf[end] after sorting
            if Xf[end - 1] <= Xf[start] + EPSILON_FLT:
                # Swap n_invalid and f_i
                tmp = features[f_i]
                features[f_i] = features[n_invalid]
                features[n_invalid] = tmp

                # Swap n_invalid and n_invalid_drawn
                tmp = features[n_invalid_drawn]
                features[n_invalid_drawn] = features[n_invalid]
                features[n_invalid] = tmp

                n_invalid += 1
                n_invalid_drawn += 1
                visited_features += 1
                continue

            # Evaluate all splits
            self.criterion.reset()
            p = start

            while p < end:
                while p + 1 < end and Xf[p + 1] <= Xf[p] + EPSILON_FLT:
                    p += 1

                # (p + 1 >= end) or (X[samples[p + 1], current_feature] >
                #                    X[samples[p], current_feature])
                p += 1
                # (p >= end) or (X[samples[p], current_feature] >
                #                X[samples[p - 1], current_feature])

                if p < end:
                    current_pos = p

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current_pos - start) < min_samples_leaf) or
                            ((end - current_pos) < min_samples_leaf)):
                        continue

                    self.criterion.update(current_pos)
                    current_improvement = self.criterion.impurity_improvement(impurity)

                    if current_improvement > best_improvement:
                        self.criterion.children_impurity(&current_impurity_left,
                                                         &current_impurity_right)
                        best_impurity_left = current_impurity_left
                        best_impurity_right = current_impurity_right
                        best_improvement = current_improvement
                        best_pos = current_pos
                        best_feature = current_feature

                        current_threshold = (Xf[p - 1] + Xf[p]) / 2.0

                        if current_threshold == Xf[p]:
                            current_threshold = Xf[p - 1]

                        best_threshold = current_threshold

            # Count one more visited feature
            visited_features += 1

            # No valid split was ever found
            if best_pos == end:
                continue

            if visited_features >= max_features:
                break

        # Reorganize into samples[start:best_pos] + samples[best_pos:end]
        if best_pos < end:
            partition_start = start
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best_feature] <= best_threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Return values
        pos[0] = best_pos
        feature[0] = best_feature
        threshold[0] = best_threshold
        impurity_left[0] = best_impurity_left
        impurity_right[0] = best_impurity_right
        impurity_improvement[0] = best_improvement
        n_valid_features[0] = n_features - n_invalid


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef class RandomSplitter(Splitter):
    """Splitter for finding the best random split."""
    def __reduce__(self):
        return (RandomSplitter, (self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SIZE_t* pos, SIZE_t* feature,
                         double* threshold, double* impurity_left,
                         double* impurity_right, double* impurity_improvement,
                         SIZE_t* n_valid_features) nogil:
        """Find the best random split on node samples[start:end]."""
        # Draw random splits and pick the best
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        # To avoid trying to split on constant features:
        # the array `features` is such that
        # `features[n_valid_features:n_features]` corresponds to features with
        # constant values at the current depth.

        cdef SIZE_t* features = self.features
        cdef SIZE_t n_valid = n_valid_features[0]
        cdef SIZE_t n_current_valid = n_valid_features[0]

        cdef DTYPE_t* X = self.X
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef double best_impurity_left = INFINITY
        cdef double best_impurity_right = INFINITY
        cdef SIZE_t best_pos = end
        cdef SIZE_t best_feature = 0
        cdef double best_threshold = 0.
        cdef double best_improvement = -INFINITY

        cdef double current_improvement
        cdef double current_impurity
        cdef double current_impurity_left
        cdef double current_impurity_right
        cdef SIZE_t current_pos
        cdef SIZE_t current_feature
        cdef double current_threshold

        cdef SIZE_t f_idx, f_i, f_j, p, tmp
        cdef SIZE_t visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value

        cdef SIZE_t partition_start
        cdef SIZE_t partition_end

        for f_idx in range(n_valid):
            # Draw a feature at random
            f_i = n_valid - f_idx - 1
            f_j = rand_int(n_valid - f_idx, random_state)
            features[f_i], features[f_j] = features[f_j], features[f_i]

            current_feature = features[f_i]

            # Find min, max
            min_feature_value = X[X_sample_stride * samples[start] +
                                  X_fx_stride * current_feature]
            max_feature_value = min_feature_value

            for p in range(start + 1, end):
                current_feature_value = X[X_sample_stride * samples[p] +
                                          X_fx_stride * current_feature]

                if current_feature_value < min_feature_value:
                    min_feature_value = current_feature_value
                elif current_feature_value > max_feature_value:
                    max_feature_value = current_feature_value

            if min_feature_value == max_feature_value:
                n_current_valid -= 1
                tmp = features[n_current_valid]
                features[n_current_valid] = current_feature
                features[f_i] = tmp
                continue

            # Draw a random threshold
            current_threshold = (min_feature_value +
                                 rand_double(random_state) *
                                 (max_feature_value - min_feature_value))

            if current_threshold == max_feature_value:
                current_threshold = min_feature_value

            # Partition
            partition_start = start
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * current_feature] <= current_threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

            current_pos = partition_end

            # Reject if min_samples_leaf is not guaranteed
            if (((current_pos - start) < min_samples_leaf) or
                    ((end - current_pos) < min_samples_leaf)):
                continue

            # Evaluate split
            self.criterion.reset()
            self.criterion.update(current_pos)
            current_improvement = self.criterion.impurity_improvement(impurity)

            if current_improvement > best_improvement:
                self.criterion.children_impurity(&current_impurity_left,
                                                 &current_impurity_right)
                best_impurity_left = current_impurity_left
                best_impurity_right = current_impurity_right
                best_improvement = current_improvement
                best_pos = current_pos
                best_feature = current_feature
                best_threshold = current_threshold

            # Count one more visited feature
            visited_features += 1

            if visited_features >= max_features:
                break

        # Reorganize into samples[start:best_pos] + samples[best_pos:end]
        if best_pos < end and current_feature != best_feature:
            partition_start = start
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best_feature] <= best_threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Return values
        pos[0] = best_pos
        feature[0] = best_feature
        threshold[0] = best_threshold
        impurity_left[0] = best_impurity_left
        impurity_right[0] = best_impurity_right
        impurity_improvement[0] = best_improvement
        n_valid_features[0] = n_current_valid


cdef class PresortBestSplitter(Splitter):
    """Splitter for finding the best split, using presorting."""
    cdef DTYPE_t* X_old
    cdef np.ndarray X_argsorted
    cdef INT32_t* X_argsorted_ptr
    cdef SIZE_t X_argsorted_stride

    cdef SIZE_t n_total_samples
    cdef unsigned char* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, object random_state):
        # Initialize pointers
        self.X_old = NULL
        self.X_argsorted_ptr = NULL
        self.X_argsorted_stride = 0
        self.sample_mask = NULL

    def __dealloc__(self):
        """Destructor."""
        free(self.sample_mask)

    def __reduce__(self):
        return (PresortBestSplitter, (self.criterion,
                                      self.max_features,
                                      self.min_samples_leaf,
                                      self.random_state), self.__getstate__())

    cdef void init(self, np.ndarray[DTYPE_t, ndim=2] X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight):
        cdef void* sample_mask = NULL

        # Call parent initializer
        Splitter.init(self, X, y, sample_weight)

        # Pre-sort X
        if self.X_old != self.X:
            self.X_old = self.X
            self.X_argsorted = np.asfortranarray(np.argsort(X, axis=0),
                                                 dtype=np.int32)

            self.X_argsorted_ptr = <INT32_t*> self.X_argsorted.data
            self.X_argsorted_stride = (<SIZE_t> self.X_argsorted.strides[1] /
                                       <SIZE_t> self.X_argsorted.itemsize)

            self.n_total_samples = X.shape[0]
            sample_mask = realloc(self.sample_mask, self.n_total_samples)
            if sample_mask == NULL:
                raise MemoryError("failed to allocate %d bytes"
                                  % self.n_total_samples)
            memset(sample_mask, 0, self.n_total_samples)
            self.sample_mask = <unsigned char*> sample_mask

    cdef void node_split(self, double impurity, SIZE_t* pos, SIZE_t* feature,
                         double* threshold, double* impurity_left,
                         double* impurity_right, double* impurity_improvement,
                         SIZE_t* n_valid_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        # To avoid trying to split on constant features:
        # the array `features` is such that
        # `features[:n_invalid]` corresponds to features with
        # constant values at the current depth,
        # `features[:n_invalid_drawn]` corresponds to features that are
        # constant and were randomly drawn by the shuffling algorithm.
        cdef SIZE_t* features = self.features
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_invalid = n_features - n_valid_features[0]
        cdef SIZE_t n_invalid_drawn = 0

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef INT32_t* X_argsorted = self.X_argsorted_ptr
        cdef SIZE_t X_argsorted_stride = self.X_argsorted_stride
        cdef SIZE_t n_total_samples = self.n_total_samples
        cdef unsigned char* sample_mask = self.sample_mask

        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef double best_impurity_left = INFINITY
        cdef double best_impurity_right = INFINITY
        cdef SIZE_t best_pos = end
        cdef SIZE_t best_feature = 0
        cdef double best_threshold = 0.
        cdef double best_improvement = -INFINITY

        cdef double current_improvement
        cdef double current_impurity
        cdef double current_impurity_left
        cdef double current_impurity_right
        cdef SIZE_t current_pos
        cdef SIZE_t current_feature
        cdef double current_threshold

        cdef SIZE_t f_idx, f_i, f_j, p, tmp
        cdef SIZE_t visited_features = 0

        cdef SIZE_t partition_start
        cdef SIZE_t partition_end

        cdef SIZE_t i, j

        # Set sample mask
        for p in range(start, end):
            sample_mask[samples[p]] = 1

        # Look for splits
        for f_idx in range(n_features):
            # Draw a feature at random
            f_i = n_invalid_drawn + n_features - f_idx - 1
            f_j = n_invalid_drawn + rand_int(n_features - f_idx, random_state)

            if f_j < n_invalid:
                # Swap f_j and n_invalid_drawn
                tmp = features[f_j]
                features[f_j] = features[n_invalid_drawn]
                features[n_invalid_drawn] = tmp
                n_invalid_drawn += 1
                visited_features += 1
                continue

            features[f_i], features[f_j] = features[f_j], features[f_i]
            current_feature = features[f_i]

            # Extract ordering from X_argsorted
            p = start

            for i in range(n_total_samples):
                j = X_argsorted[X_argsorted_stride * current_feature + i]
                if sample_mask[j] == 1:
                    samples[p] = j
                    Xf[p] = X[X_sample_stride * j +
                              X_fx_stride * current_feature]
                    p += 1

            # The feature is constant since Xf[start] ~ Xf[end] after sorting
            if Xf[end - 1] <= Xf[start] + EPSILON_FLT:
                # Swap n_invalid and f_i
                tmp = features[f_i]
                features[f_i] = features[n_invalid]
                features[n_invalid] = tmp

                # Swap n_invalid and n_invalid_drawn
                tmp = features[n_invalid_drawn]
                features[n_invalid_drawn] = features[n_invalid]
                features[n_invalid] = tmp

                n_invalid += 1
                n_invalid_drawn += 1
                visited_features += 1
                continue

            # Evaluate all splits
            self.criterion.reset()
            p = start

            while p < end:
                while p + 1 < end and Xf[p + 1] <= Xf[p] + EPSILON_FLT:
                    p += 1

                # (p + 1 >= end) or (X[samples[p + 1], current_feature] >
                #                    X[samples[p], current_feature])
                p += 1
                # (p >= end) or (X[samples[p], current_feature] >
                #                X[samples[p - 1], current_feature])

                if p < end:
                    current_pos = p

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current_pos - start) < min_samples_leaf) or
                            ((end - current_pos) < min_samples_leaf)):
                        continue

                    self.criterion.update(current_pos)
                    current_improvement = self.criterion.impurity_improvement(impurity)

                    if current_improvement > best_improvement:
                        self.criterion.children_impurity(&current_impurity_left,
                                                         &current_impurity_right)
                        best_impurity_left = current_impurity_left
                        best_impurity_right = current_impurity_right
                        best_improvement = current_improvement
                        best_pos = current_pos
                        best_feature = current_feature

                        current_threshold = (Xf[p - 1] + Xf[p]) / 2.0

                        if current_threshold == Xf[p]:
                            current_threshold = Xf[p - 1]

                        best_threshold = current_threshold

            # Count one more visited feature
            visited_features += 1

            # No valid split was ever found
            if best_pos == end:
                continue

            if visited_features >= max_features:
                break

        # Reorganize into samples[start:best_pos] + samples[best_pos:end]
        if best_pos < end:
            partition_start = start
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p]
                     + X_fx_stride * best_feature] <= best_threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Reset sample mask
        for p in range(start, end):
            sample_mask[samples[p]] = 0

        # Return values
        pos[0] = best_pos
        feature[0] = best_feature
        threshold[0] = best_threshold
        impurity_left[0] = best_impurity_left
        impurity_right[0] = best_impurity_right
        impurity_improvement[0] = best_improvement
        n_valid_features[0] = n_features - n_invalid


# =============================================================================
# Tree builders
# =============================================================================
cdef class TreeBuilder:
    """Interface for different tree building strategies. """

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        pass


# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        # check if dtype is correct
        if X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            if (sample_weight.dtype != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.asarray(sample_weight,
                                           dtype=DOUBLE, order="C")
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Recursive partition (without actual recursion)
        cdef Splitter splitter = tree.splitter
        splitter.init(X, y, sample_weight_ptr)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t n_valid_features = splitter.n_features
        cdef SIZE_t pos
        cdef SIZE_t feature
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef double split_impurity_left = INFINITY
        cdef double split_impurity_right = INFINITY
        cdef double split_improvement = INFINITY
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        # push root node onto stack
        rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY,
                        n_valid_features)
        if rc == -1:
            # got return code -1 - out-of-memory
            raise MemoryError()

        with nogil:
            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_valid_features = stack_record.n_valid_features

                n_node_samples = end - start
                is_leaf = ((depth >= tree.max_depth) or
                           (n_node_samples < tree.min_samples_split) or
                           (n_node_samples < 2 * tree.min_samples_leaf))

                splitter.node_reset(start, end)  # calls criterion.init

                if first:
                    impurity = splitter.criterion.node_impurity()
                    first = 0

                is_leaf = is_leaf or (impurity < EPSILON_FLT)

                if not is_leaf:
                    splitter.node_split(impurity, &pos, &feature, &threshold,
                                        &split_impurity_left,
                                        &split_impurity_right,
                                        &split_improvement,
                                        &n_valid_features)
                    is_leaf = is_leaf or (pos >= end)

                node_id = tree._add_node(parent, is_left, is_leaf, feature,
                                         threshold, impurity, n_node_samples)

                if is_leaf:
                    # Don't store value for internal nodes
                    splitter.node_value(tree.value
                                        + node_id * tree.value_stride)

                else:
                    # Push right child on stack
                    rc = stack.push(pos, end, depth + 1, node_id, 0,
                                    split_impurity_right, n_valid_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, pos, depth + 1, node_id, 1,
                                    split_impurity_left, n_valid_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)
            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

        # release memory
        tree.splitter = None


# Best first builder ----------------------------------------------------------

cdef int _add_split_node(Splitter splitter, Tree tree,
                         SIZE_t start, SIZE_t end, double impurity,
                         bint is_first, bint is_left, Node* parent,
                         SIZE_t depth, PriorityHeapRecord* res) nogil:
    """Adds node w/ partition ``[start, end)`` to the frontier. """
    cdef SIZE_t pos
    cdef SIZE_t feature
    cdef SIZE_t node_id
    cdef SIZE_t n_valid_features = splitter.n_features
    cdef double threshold
    cdef double split_impurity_left
    cdef double split_impurity_right
    cdef double split_improvement
    cdef SIZE_t n_node_samples
    cdef bint is_leaf
    cdef SIZE_t n_left, n_right
    cdef double imp_diff

    splitter.node_reset(start, end)  # calls criterion.init
    if is_first:
        impurity = splitter.criterion.node_impurity()

    n_node_samples = end - start
    is_leaf = ((depth > tree.max_depth) or
               (n_node_samples < tree.min_samples_split) or
               (n_node_samples < 2 * tree.min_samples_leaf))

    if not is_leaf:
        splitter.node_split(impurity, &pos, &feature, &threshold,
                            &split_impurity_left, &split_impurity_right,
                            &split_improvement, &n_valid_features)
        is_leaf = is_leaf or (pos >= end)

    node_id = tree._add_node(parent - tree.nodes if parent != NULL
                             else _TREE_UNDEFINED,
                             is_left, is_leaf,
                             feature, threshold, impurity, n_node_samples)
    if node_id == <SIZE_t>(-1):
        return -1

    # compute values also for split nodes (might become leafs later).
    splitter.node_value(tree.value + node_id * tree.value_stride)

    res.node_id = node_id
    res.start = start
    res.end = end
    res.depth = depth
    res.impurity = impurity

    if not is_leaf:
        # is split node
        res.pos = pos
        res.is_leaf = 0
        res.improvement = split_improvement
    else:
        # is leaf => 0 improvement
        res.pos = end
        res.is_leaf = 1
        res.improvement = 0.0

    return 0


cdef int _add_to_frontier(PriorityHeapRecord* rec, PriorityHeap frontier) nogil:
    """Adds record ``rec`` to the priority queue ``frontier``; returns -1 on
       memory-error."""
    return frontier.push(rec.node_id, rec.start, rec.end, rec.pos, rec.depth,
                         rec.is_leaf, rec.improvement, rec.impurity)


cdef class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.

    NOTE: this TreeBuilder will ignore ``tree.max_depth`` .
    """

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        # Check if dtype is correct
        if X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            if (sample_weight.dtype != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.asarray(sample_weight,
                                           dtype=DOUBLE, order="C")
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Recursive partition (without actual recursion)
        cdef Splitter splitter = tree.splitter
        splitter.init(X, y, sample_weight_ptr)

        cdef PriorityHeap frontier = PriorityHeap(INITIAL_STACK_SIZE)
        cdef PriorityHeapRecord record
        cdef PriorityHeapRecord split_node_left
        cdef PriorityHeapRecord split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef int max_leaf_nodes = tree.max_leaf_nodes
        cdef int max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        # Initial capacity
        cdef int init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = _add_split_node(splitter, tree,
                                 0, n_node_samples, INFINITY, IS_FIRST,
                                 IS_LEFT, NULL, 0, &split_node_left)
            if rc >= 0:
                rc = _add_to_frontier(&split_node_left, frontier)
        if rc == -1:
            raise MemoryError()

        with nogil:
            while not frontier.is_empty():
                frontier.pop(&record)

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    rc = _add_split_node(splitter, tree,
                                         record.start, record.pos,
                                         record.impurity, IS_NOT_FIRST,
                                         IS_LEFT, node, record.depth + 1,
                                         &split_node_left)
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    rc = _add_split_node(splitter, tree, record.pos,
                                         record.end, record.impurity,
                                         IS_NOT_FIRST, IS_NOT_LEFT, node,
                                         record.depth + 1, &split_node_right)
                    if rc == -1:
                        break

                    # Add nodes to queue
                    rc = _add_to_frontier(&split_node_left, frontier)
                    if rc == -1:
                        break

                    rc = _add_to_frontier(&split_node_right, frontier)
                    if rc == -1:
                        break

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)
            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

        # release memory
        tree.splitter = None


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, ]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_samples[i] holds the number of training samples reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            # it's small; copy for memory safety
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs).copy()

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs, Splitter splitter, SIZE_t max_depth,
                  SIZE_t min_samples_split, SIZE_t min_samples_leaf,
                  int max_leaf_nodes, object random_state):
        """Constructor."""
        # Inner structures
        self.node_count = 0
        self.capacity = 0
        self.value = NULL

        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = <SIZE_t*> malloc(n_outputs * sizeof(SIZE_t))

        if self.n_classes == NULL:
            raise MemoryError()

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Parameters
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

        # Inner structures
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs,
                       self.splitter,
                       self.max_depth,
                       self.min_samples_split,
                       self.min_samples_leaf,
                       self.max_leaf_nodes,
                       self.random_state), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs, self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        self._resize_c(self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef void _resize(self, SIZE_t capacity):
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays."""
        if self._resize_c(capacity)!= 0:
            raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
        """Guts of _resize. Returns 0 for success, -1 for error."""
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        ptr = realloc(self.value,
                      capacity * self.value_stride * sizeof(double))
        if ptr == NULL:
            return -1
        self.value = <double*> ptr

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(((<void*> self.value) + self.capacity *
                    self.value_stride * sizeof(double)), 0,
                   ((capacity - self.capacity) * self.value_stride *
                    sizeof(double)))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples) nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_samples = n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, np.ndarray[DTYPE_t, ndim=2] X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        cdef SIZE_t n_samples = X.shape[0]
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_data = <SIZE_t*> out.data

        with nogil:
            for i in range(n_samples):
                node = self.nodes

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_data[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef SIZE_t n_left
        cdef SIZE_t n_right
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))

        while node != end_node:
            if node.left_child != _TREE_LEAF:
                # ... and node.right_child != _TREE_LEAF:
                left = &nodes[node.left_child]
                right = &nodes[node.right_child]
                n_left = left.n_samples
                n_right = right.n_samples

                importances[node.feature] += (node.n_samples * node.impurity -
                                              left.n_samples * left.impurity -
                                              right.n_samples * right.impurity)
            node += 1

        importances = importances / nodes[0].n_samples
        cdef double normalizer

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr


# =============================================================================
# Utils
# =============================================================================

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)

cdef inline SIZE_t rand_int(SIZE_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef inline double rand_double(UINT32_t* random_state) nogil:
    """Generate a random double in [0; 1)."""
    return <double> our_rand_r(random_state) / <double> RAND_R_MAX

cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)
