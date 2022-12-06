""" module of functions related to discovering feature axis """

import time
import numpy as np
import sklearn.linear_model as linear_model
from math import copysign, hypot


def find_feature_axis(z, y, method='linear', **kwargs_model):
    """
    function to find axis in the latent space that is predictive of feature vectors

    :param z: vectors in the latent space, shape=(num_samples, num_latent_vector_dimension)
    :param y: feature vectors, shape=(num_samples, num_features)
    :param method: one of ['linear', 'logistic'], or a sklearn.linear_model object, (eg. sklearn.linear_model.ElasticNet)
    :param kwargs_model: parameters specific to a sklearn.linear_model object, (eg., penalty=’l2’)
    :return: feature vectors, shape = (num_latent_vector_dimension, num_features)
    """

    if method == 'linear':
        model = linear_model.LinearRegression(**kwargs_model)
        model.fit(z, y)
    elif method == 'tanh':
        def arctanh_clip(y):
            return np.arctanh(np.clip(y, np.tanh(-3), np.tanh(3)))

        model = linear_model.LinearRegression(**kwargs_model)

        model.fit(z, arctanh_clip(y))
    else:
        raise Exception('method has to be one of ["linear", "tanh"]')

    return model.coef_.transpose()
    #输出权重的转置


def normalize_feature_axis(feature_slope):
    """
    function to normalize the slope of features axis so that they have the same length

    :param feature_slope: array of feature axis, shape = (num_latent_vector_dimension, num_features)
    :return: same shape of input
    """

    feature_direction = feature_slope / np.linalg.norm(feature_slope, ord=2, axis=0, keepdims=True)
    return feature_direction


def disentangle_feature_axis(feature_axis_target, feature_axis_base, yn_base_orthogonalized=False):
    """
    make feature_axis_target orthogonal to feature_axis_base

    :param feature_axis_target: features axes to decorrerelate, shape = (num_dim, num_feature_0)
    :param feature_axis_base: features axes to decorrerelate, shape = (num_dim, num_feature_1))
    :param yn_base_orthogonalized: True/False whether the feature_axis_base is already othogonalized
    :return: feature_axis_decorrelated, shape = shape = (num_dim, num_feature_0)
    """

    # make sure this funciton works to 1D vector 此处代码0可能有问题，仅此出的有问题
    if len(feature_axis_target.shape) == 0:
        yn_single_vector_in = True
        feature_axis_target = feature_axis_target[:, None] #把这个目标向量转换成一个 1D vector 转换成一个列向量
    else:
        yn_single_vector_in = False

    # if already othogonalized, skip this step 这一步似乎有问题，上下两个函数应该要换一下
    #主要目的是把base里面的所有向量调成两两垂直的模式
    if yn_base_orthogonalized:
        feature_axis_base_orthononal = orthogonalize_vectors(feature_axis_base)
    else:
        feature_axis_base_orthononal = feature_axis_base

    # orthogonalize every vector
    #此处加0是起什么作用？此步骤的作用是调整target中的每一列的vector使其能够和base里面的每个vector垂直
    feature_axis_decorrelated = feature_axis_target + 0
    num_dim, num_feature_0 = feature_axis_target.shape
    num_dim, num_feature_1 = feature_axis_base_orthononal.shape
    for i in range(num_feature_0):
        for j in range(num_feature_1):
            feature_axis_decorrelated[:, i] = orthogonalize_one_vector(feature_axis_decorrelated[:, i],
                                                                       feature_axis_base_orthononal[:, j])

    # make sure this funciton works to 1D vector
    #依然是转换成一列的向量
    if yn_single_vector_in:
        result = feature_axis_decorrelated[:, 0]
    else:
        result = feature_axis_decorrelated

    return result


def disentangle_feature_axis_by_idx(feature_axis, idx_base=None, idx_target=None, yn_normalize=True):
    """
    disentangle correlated feature axis, make the features with index idx_target orthogonal to
    those with index idx_target, wrapper of function disentangle_feature_axis()

    :param feature_axis:       all features axis, shape = (num_dim, num_feature)
    :param idx_base:           index of base features (1D numpy array), to which the other features will be orthogonal
    :param idx_target: index of features to disentangle (1D numpy array), which will be disentangled from
                                    base features, default to all remaining features
    :param yn_normalize:       True/False to normalize the results
    :return:                   disentangled features, shape = feature_axis
    """

    (num_dim, num_feature) = feature_axis.shape

    # process default input
    if idx_base is None or len(idx_base) == 0:    # if None or empty, do nothing
        feature_axis_disentangled = feature_axis
    else:                                         # otherwise, disentangle features
        if idx_target is None:                # if None, use all remaining features
            idx_target = np.setdiff1d(np.arange(num_feature), idx_base)
        #从下面的feature name用0～n命名可知，此处np.arange()相当于把所有的feature列出来，然后和idx_base里面所包含都有的feature做比较，从而找出idx_base里面缺少哪些features.
        #setdiff1d用于从输入一中找出输入二中不包含的内容，作为traget feature
        feature_axis_target = feature_axis[:, idx_target] + 0
        feature_axis_base = feature_axis[:, idx_base] + 0
        feature_axis_base_orthogonalized = orthogonalize_vectors(feature_axis_base)
        feature_axis_target_orthogonalized = disentangle_feature_axis(
            feature_axis_target, feature_axis_base_orthogonalized, yn_base_orthogonalized=True)
            #此处调用了上方的那个函数

        feature_axis_disentangled = feature_axis + 0  # holder of results
        feature_axis_disentangled[:, idx_target] = feature_axis_target_orthogonalized
        feature_axis_disentangled[:, idx_base] = feature_axis_base_orthogonalized

    # normalize output
    if yn_normalize:
        feature_axis_out = normalize_feature_axis(feature_axis_disentangled)
    else:
        feature_axis_out = feature_axis_disentangled
    return feature_axis_out

def new_disentangle_feature_axis_by_idx(feature_axis):
    feature_axis_out = givens_reduce(feature_axis)
    return feature_axis_out

def householder_reduce(matrix_a):
    """
    Householder reduce for QR factorization.
    :param matrix_a:
    :return: matrix_q, matrix_r
    """

    # row num: n
    # col num: m
    n, m = matrix_a.shape

    matrix_r = matrix_a

    # H_1, H_2, ..., H_n
    householder_matrix_list = []

    # project each col onto standard basis
    for j in range(m):

        # Deal with un-reduced sub-matrix.
        sub_matrix = matrix_r[j:, j:]

        """ Get a_j """
        # a_j: column vector j
        a = np.reshape(sub_matrix[:, 0],
                       (len(sub_matrix), 1))

        # Check j-col
        if not np.nonzero(a)[0].any() or len(a) == 1:
            # All rested elements in col_j are zeros.
            continue

        """ Get v_j = a - |a| e """

        # 2-norm of vector a
        a_norm_2 = np.int(np.sqrt(np.matmul(a.T, a)))

        # standard base e
        e = np.zeros_like(a)
        e[0] = 1

        # v = a - |a| e
        v = np.subtract(a, a_norm_2 * e)

        """ Get Householder matrix H_j"""

        # Household matrix H: I - 2 (vv')/(v'v)
        sub_matrix_h = np.identity(len(v)) - 2 * np.matmul(v, v.T) / np.matmul(v.T, v)

        # Augment Household matrix
        matrix_h = np.identity(n)
        matrix_h[j:, j:] = sub_matrix_h

        # Mapping current matrix
        matrix_r = np.matmul(matrix_h, matrix_r)

        # Store Household matrix
        householder_matrix_list.append(matrix_h)

    """ Reduce R matrix"""
    matrix_r = matrix_r[0:m]

    """ Compute Q' matrix """
    # Compute Q', where Q' = H_n ... H_2 * H_1 * I
    matrix_q = np.identity(n)

    for household_matrix in householder_matrix_list:
        matrix_q = np.matmul(household_matrix, matrix_q)

    """ Reduce Q matrix """
    matrix_q = np.transpose(matrix_q[0:m])

    return matrix_q

def givens_reduce(matrix_a):
    r"""
    Givens reduce for QR factorization.
    :param matrix_a:
    :return: matrix_q, matrix,r

    Parameters:
    -----------
    matrix_a: np.ndarray
        original matrix to be Givens Reduce factorized.

    """

    # row num: n
    # col num: m
    n, m = matrix_a.shape

    matrix_r = matrix_a

    # R_1, R_2, ... R_n
    givens_matrix_list = []

    # Rotation each entry in matrix
    for j in range(m):  # Col-m
        for i in range(j+1, n):  # Row-n

            if matrix_r[i][j] == 0:
                continue

            """ Find a and b (current entry) """
            a = matrix_r[j][j]
            b = matrix_r[i][j]

            # Prepare c and s
            base = np.sqrt(np.power(a, 2) + np.power(b, 2))
            c = np.true_divide(a, base)
            s = np.true_divide(b, base)

            # Givens trans matrix
            matrix_g = np.identity(n)

            matrix_g[j][j] = c  # Upper Left
            matrix_g[i][j] = -s  # Lower Left

            matrix_g[j][i] = s  # Upper Right
            matrix_g[i][i] = c  # Lower Right

            # Rotation
            matrix_r = np.matmul(matrix_g, matrix_r)

            givens_matrix_list.append(matrix_g)

    """ Reduce R matrix """
    matrix_r = matrix_r[0: m]

    # Compute Q', where Q' = Rn...R2*R1*A
    matrix_q = np.identity(n)
    for givens_matrix in givens_matrix_list:
        matrix_q = np.matmul(givens_matrix, matrix_q)

    # Get Q
    matrix_q = np.transpose(matrix_q[0: m])

    return matrix_q

def orthogonalize_one_vector(vector, vector_base):
    #此处将一个向量投影到垂直于另一个向量的方向上，仍然是用相减的方法，会造成较大的误差，此处是可更改的点
    """
    tool function, adjust vector so that it is orthogonal to vector_base (i.e., vector - its_projection_on_vector_base )

    :param vector0: 1D array
    :param vector1: 1D array
    :return: adjusted vector1
    """
    return vector - np.dot(vector, vector_base) / np.dot(vector_base, vector_base) * vector_base


def orthogonalize_vectors(vectors):
    #此处是对base_vector_space的所有vector做正交化，方法和加入新的target vector所用的正交化的方法相同
    """
    tool function, adjust vectors so that they are orthogonal to each other, takes O(num_vector^2) time

    :param vectors: vectors, shape = (num_dimension, num_vector)
    :return: orthorgonal vectors, shape = (num_dimension, num_vector)
    """
    vectors_orthogonal = vectors + 0
    num_dimension, num_vector = vectors.shape
    for i in range(num_vector):
        for j in range(i):
            vectors_orthogonal[:, i] = orthogonalize_one_vector(vectors_orthogonal[:, i], vectors_orthogonal[:, j])
    return vectors_orthogonal


def plot_feature_correlation(feature_direction, feature_name=None):
    import matplotlib.pyplot as plt

    len_z, len_y = feature_direction.shape
    if feature_name is None:
        feature_name = range(len_y)
    #特征名称是0～n，此处计算该矩阵中所有方向的相关系数，相关系数矩阵如果除了斜对角线是0，其他都是0的话，是最好的情况,但是此处仅用于画图，可以通过相关系数的求解来做一些微调
    feature_correlation = np.corrcoef(feature_direction.transpose())

    c_lim_abs = np.max(np.abs(feature_correlation))

    plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_correlation,
                   cmap='coolwarm', vmin=-c_lim_abs, vmax=+c_lim_abs)
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.axis('square')
    plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
    plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
    plt.show()


def plot_feature_cos_sim(feature_direction, feature_name=None):
    """
    plot cosine similarity measure of vectors

    :param feature_direction: vectors, shape = (num_dimension, num_vector)
    :param feature_name:      list of names of features
    :return:                  cosines similarity matrix, shape = (num_vector, num_vector)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    len_z, len_y = feature_direction.shape
    if feature_name is None:
        feature_name = range(len_y)

    feature_cos_sim = cosine_similarity(feature_direction.transpose())

    c_lim_abs = np.max(np.abs(feature_cos_sim))

    plt.pcolormesh(np.arange(len_y+1), np.arange(len_y+1), feature_cos_sim,
                   vmin=-c_lim_abs, vmax=+c_lim_abs, cmap='coolwarm')
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.axis('square')
    plt.xticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small', rotation='vertical')
    plt.yticks(np.arange(len_y) + 0.5, feature_name, fontsize='x-small')
    plt.show()
    return feature_cos_sim



