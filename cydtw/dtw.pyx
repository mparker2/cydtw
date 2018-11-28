cimport cython
import numpy as np
cimport numpy as np


NP_FLOAT = np.float
NP_INT = np.int


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cityblock_cdist(np.ndarray X, np.ndarray Y):
    '''
    Compute distance matrix between two signals 
    using cityblock i.e. absolute distance
    '''
    cdef int i, j
    cdef np.ndarray dm
    dm = np.empty((X.size, Y.size), dtype=NP_FLOAT)
    for i in range(X.size):
        for j in range(Y.size):
            dm[i, j] = abs(X[i] - Y[j])
    return dm


@cython.boundscheck(False)
@cython.wraparound(False)
cdef accum_cost(np.ndarray cost):
    '''
    build accumulated cost matrix from distance matrix
    '''
    cdef int i, j, imax, jmax, step_type
    cdef int skipi, skipj
    cdef float pos_cost, step_cost, step_cost_skipi, step_cost_skipj
    cdef np.ndarray ij, accum_cost, path_matrix
    imax = cost.shape[0] + 1
    jmax = cost.shape[1] + 1
    accum_cost = np.full(shape=(imax, jmax),
                         fill_value=np.inf,
                         dtype=NP_FLOAT)
    accum_cost[0, 0] = 0
    path_matrix = np.zeros(shape=(imax, jmax), dtype=NP_INT)
    for i in range(1, imax):
        for j in range(1, jmax):
            pos_cost = cost[i - 1, j - 1]
            step_type = 0
            step_cost = accum_cost[i - 1, j - 1]
            step_cost_skipi = accum_cost[i - 1, j]
            step_cost_skipj = accum_cost[i, j - 1]
            skipi = step_cost_skipi < step_cost
            skipj = step_cost_skipj < step_cost
            if skipi:
                if skipj:
                    if step_cost_skipi < step_cost_skipi:
                        step_cost = step_cost_skipi
                        step_type = 1
                    else:
                        step_cost = step_cost_skipj
                        step_type = 2
                else:
                    step_cost = step_cost_skipi
                    step_type = 1
            elif skipj:
                step_cost = step_cost_skipj
                step_type = 2
            path_matrix[i, j] = step_type
            accum_cost[i, j] = pos_cost + step_cost
    return accum_cost[1:, 1:], path_matrix[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef backtrack_path_matrix(np.ndarray path_matrix):
    '''
    Find optimal alignment of sequences by backtracking
    through accumulated cost matrix
    '''
    cdef int i, j, n
    cdef np.ndarray path
    n = 0
    i = path_matrix.shape[0] - 1
    j = path_matrix.shape[1] - 1
    # longest possible backtrack would be i + j + 1
    path = np.empty(shape=(i + j + 1, 2), dtype=NP_INT)
    path[n, 0] = i
    path[n, 1] = j
    while i != 0 or j != 0:
        n += 1
        p = path_matrix[i, j]
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        elif p == 2:
            j -= 1
        path[n, 0] = i
        path[n, 1] = j
    return path[:n + 1]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dtw(np.ndarray X, np.ndarray Y, bint backtrack=False):
    '''
    Compute the Dynamic Time Warp cost matrix between
    two sequences, and optionally backtrack to find
    optimal alignment between the sequences

    
    Parameters
    ----------
    
    X: np.ndarray, shape M, required
        Reference signal to perform DTW alignment on

    Y: np.ndarray, shape M, required
        Query signal to perform DTW alignment on

    backtrack: bool, optional, default: False
        Whether or not to perform backtracking through
        the accumulated cost matrix, to produce an
        alignment between the two input signals

    Returns
    -------

    accum: np.ndarray, shape (M, N)
        The accumulated cost matrix for the alignment
        of X and Y. The overall cost of alignment is
        accum[-1, -1]

    path: np.ndarray, shape (P, 2)
        The optimal alignment path for X and Y
    '''
    assert X.ndim == 1
    assert Y.ndim == 1
    cdef np.ndarray c, accum, pmat, path
    c = cityblock_cdist(X, Y)
    accum, pmat = accum_cost(c)
    if backtrack:
        path = backtrack_path_matrix(pmat)
        return accum, path
    else:
        return accum


@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_pdist(np.ndarray X):
    '''
    Compute the pairwise euclidean distance matrix
    between a set of signals
    '''
    cdef int i, j, eu
    cdef int n_seqs = len(X)
    cdef np.ndarray dm
    dm = np.empty((n_seqs, n_seqs), dtype=NP_FLOAT)
    for i in range(n_seqs):
        for j in range(i):
            eu = np.sum((X[i] - X[j]) ** 2)
            dm[i, j] = eu
            dm[j, i] = eu
        dm[i, i] = 0
    return np.sqrt(dm)
        

@cython.boundscheck(False)
@cython.wraparound(False)
cdef medoid(np.ndarray X):
    '''
    Find the medoid of a set of signals
    '''
    cdef np.ndarray dist, dist_sum, m
    dists = euclidean_pdist(X)
    dist_sum = dists.sum(0)
    m = X[dist_sum.argmin()]
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _update_dba(np.ndarray t, np.ndarray X):
    '''
    Perform one iteration of DBA
    '''
    cdef np.ndarray x, t_update, t_count, path
    t_update = np.zeros_like(t)
    t_count = np.zeros_like(t)
    for x in X:
        _, path = dtw(t, x, backtrack=True)
        for i, j in path:
            t_update[i] += x[j]
            t_count[i] += 1
    t_update = t_update / t_count
    return np.asarray(t_update)


@cython.boundscheck(False)
@cython.wraparound(False)
def dba(np.ndarray X, int n_iter=20, float tol=0.05):
    '''
    Compute the DTW Barycenter Average of a set of
    signals.
    
    Parameters
    ----------
    
    X: np.ndarray, shape (M, N), required
        The set of M signals of length N, to perform DBA on

    n_iter: int, optional, default: 20
        The max number of DBA iterations to perform

    tol: float, optional, default: 0.05
        The tolerance at which to consider the average signal
        to be converged.

    Returns
    -------

    t: np.ndarray, shape N
        The DBA average signal
    '''
    cdef np.ndarray t, t_update
    t = medoid(X)
    for _ in range(n_iter):
        t_update = _update_dba(t, X)
        if np.allclose(t, t_update, 0.05):
            return t_update
        else:
            t = t_update
    return t
