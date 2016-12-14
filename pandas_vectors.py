import collections

_vector_lists = dict(
    xyz=['_x', '_y', '_z'],
    xy=['_x', '_y'],
    pyr=['_p', '_y', '_r'],
    PYR=['_pitch', '_yaw', '_roll'],
)

class _VectorNames():
    def __init__(self, veclist):
        if not isinstance(veclist, collections.Hashable):
            # TODO: check that its valid?
            self._veclist = veclist
            return
        if veclist in _vector_lists.keys():
            self._veclist = _vector_lists[veclist]
            return

    def __enter__(self):
        self._orig = _veclist
        set_vectornames(self._veclist)

    def __exit__(self, type, value, traceback):
        set_vectornames(self._orig)


_veclist = ['_x', '_y', '_z']


def vectornames(vector_list):
    return _VectorNames(vector_list)


def set_vectornames(vector_list):
    vn = _VectorNames(vector_list)
    global _veclist
    _veclist = vn._veclist


def _vector_to_list(vectors):
    """ Convenience function to convert string to list to preserve iterables. """
    if isinstance(vectors, str):
        vectors = [vectors]
    return vectors


def slice(df, vectors):
    """ Return a dataframe slice of the vector """
    return df.loc[:, indexer(vectors)]


def indexer(vectors):
    """
    Return a list of the vector names for each vector in vectors.

    Essentially expands the vectors list to add the vector suffix to each.
    """
    vectors = _vector_to_list(vectors)
    return [vector + xyz for vector in vectors for xyz in _veclist]


def all_indexer(df):
    """ Return indexer for all vector columns in dataframe. """
    columns = [df.columns.str.endswith(xyz) for xyz in _veclist]
    vector_columns = columns[0]
    for column in columns:
        vector_columns |= column
    return df.columns[vector_columns]


def all_vectors(df):
    """
    Return a list of all the vector names of in dataframe.
    Assumes that anything that ends in the first suffix is a vector.
    """
    vectors = df.columns[df.columns.str.endswith(_veclist[0])]
    vectors = [vector[:-len(_veclist[0])] for vector in vectors]
    return vectors


def rename(df, vectors, new):
    """ Rename vectors to new names. List of vectors maps to new one-to-one. """
    vectors = _vector_to_list(vectors)
    new = _vector_to_list(new)
    if len(vectors) != len(new):
        raise ValueError("length of lists must match")
    old = indexer(vectors)
    new = indexer(new)
    return df.rename(columns=dict(zip(old, new)))


def copy(df, vector, new):
    """ Copy vectors to new names. List of vectors maps to new one-to-one. """
    vectors = _vector_to_list(vectors)
    new = _vector_to_list(new)
    if len(vectors) != len(new):
        raise ValueError("length of lists must match")
    df = df.copy()
    for old, new_ in zip(indexer(vectors), indexer(new)):
        df[new_] = df[old]
    return df


def copy_vectors_suffix(df, vectors, suffix):
    """
    Copy vector to new name with suffix.
    suffix can be length 1 (add suffix to all) or equal to length of vectors (mapped one-to-one).
    """
    df = df.copy()
    vectors = _vector_to_list(vectors)
    suffix = _vector_to_list(suffix)
    if len(suffix) == 1:
        suffixes = [vector + suffix for vector in vectors]
    elif len(suffix) == len(vectors):
        suffixes = [vector + suffix_ for vector, suffix_ in zip(vectors, suffix)]
    return copy(df, vectors, suffixes)


def _vector_addsubtract(df_left, vectors_left, df_right=None, vectors_right=None, subtract=False):
    if df_right is None and vectors_right is None:
        raise ValueError('Need to give one of df_right or vectors_right')
    if df_right is None:
        # add vectors within the dataframe
        return _vector_addsubtract(df_left, vectors_left, df_left, vectors_right, subtract)
    if vectors_right is None:
        # add like named vectors
        return _vector_addsubtract(df_left, vectors_left, df_right, vectors_left, subtract)

    vectors_left = _vector_to_list(vectors_left)
    vectors_right = _vector_to_list(vectors_right)

    df_left = df_left.copy()
    if len(vectors_left) == len(vectors_right):
        if subtract:
            df_left.loc[:, indexer(vectors_left)] -= df_right.loc[:, indexer(vectors_right)].values
        else:
            df_left.loc[:, indexer(vectors_left)] += df_right.loc[:, indexer(vectors_right)].values
    elif len(vectors_right) == 1:
        for vector in vectors_left:
            if subtract:
                df_left.loc[:, indexer(vector)] -= df_right.loc[:, indexer(vectors_right)].values
            else:
                df_left.loc[:, indexer(vector)] += df_right.loc[:, indexer(vectors_right)].values
    else:
        raise ValueError('vectors_right needs to be length 1 or same as vectors_left')

    return df_left


def subtract(df_left, vectors_left, df_right=None, vectors_right=None):
    """
    Subtract vectors in a dataframe or in different dataframes.
    Must provide one or both of df_right or vectors_right.
    If you provide only df_right, subtract vectors_left in df_right from vectors_left in df_left.
    If you provide only vectors_right, subtract vectors_right from vectors_left in df_left.
    If you provide both, subtract vectors_right in df_right from vectors_left in df_left.

    vectors_right needs to be length 1 or same as vectors_left.
    If vectors_right is length 1, then subtract same vector from all given in vectors_left.
    """
    return _vector_addsubtract(df_left, vectors_left, df_right, vectors_right, subtract=True)


def add(df_left, vectors_left, df_right=None, vectors_right=None):
    """
    Add vectors in a dataframe or in different dataframes.
    Must provide one or both of df_right or vectors_right.
    If you provide only df_right, add vectors_left in df_right to vectors_left in df_left.
    If you provide only vectors_right, add vectors_right to vectors_left in df_left.
    If you provide both, add vectors_right in df_right to vectors_left in df_left.

    vectors_right needs to be length 1 or same as vectors_left.
    If vectors_right is length 1, then add same vector to all given in vectors_left.
    """
    return _vector_addsubtract(df_left, vectors_left, df_right, vectors_right, subtract=False)


def matrix_mult(df, vectors, suffix, matrix, stack_ones=True):
    """
    Performs y = M*x for the vector or vectors and saves the result in
    the in df[vector + suffix]. The matrix must be square and proper dimensions.

    If stack_ones is True, this will add an extra column of ones to the vectors
    so that you can use a transformation matrix of size len(vectorlist)+1.

    If stack_ones is False, matrix must be size len(vectorlist).

    Returns the modified df as a copy of the original
    """
    import numpy as np

    df = df.copy()
    vectors = _vector_to_list(vectors)
    for vector in vectors:
        # Add empty columns for new vector names
        if vector != vector + suffix and vector + suffix + _veclist[0] not in df.columns:
            df = df.reindex(columns=df.columns.values.tolist() + indexer(vector + suffix))
        if stack_ones:
            npvecs = np.column_stack((slice(df, vector), np.ones(len(df))))
        else:
            npvecs = np.array(slice(df, vector))
        df.loc[:, indexer(vector + suffix)] = matrix.dot(npvecs.T).T[:, :len(_veclist)]
    return df


def transform(df, vectors, suffix, func, *args, **kwargs):
    """
    Run a function (func) on each column of a vector or vectors,
    saving the result in df[vector + suffix]. The result of the function
    must be the same length as the original vector.

    Returns the modified df as a copy of the original
    """
    df = df.copy()
    vectors = _vector_to_list(vectors)
    out = [vector + suffix for vector in vectors]
    for vector, out in zip(indexer(vectors), indexer(out)):
        df[out] = func(df[vector], *args, **kwargs)
    return df


def diff(df, vectors, suffix='_diff', shift=1):
    """
    Run diff on each vector column, saving the result in df[vector + suffix].
    """
    df = df.copy()
    vectors = _vector_to_list(vectors)
    out = [vector + suffix for vector in vectors]
    for vector, out in zip(indexer(vectors), indexer(out)):
        df[out] = df[vector].diff(shift)
    return df


def _magnitude(df, vector):
    """ Return the magnitude of the vector """
    import numpy as np
    return np.sqrt(np.square(slice(df, vector)).sum(axis=1))


def magnitude(df, vectors, suffix='_mag'):
    """
    Calculate the vector magnitude of each vector in vectors, saving the result in df[vector + suffix].
    """
    df = df.copy()
    vectors = _vector_to_list(vectors)
    for vector in vectors:
        df[vector + suffix] = _magnitude(df, vector)
    return df
