#!/usr/bin/env python27
"""
Module: cse6040utils

Some utility functions created for Georgia Tech's CSE 6040: Computing for Data Analysis.
"""

#============================================================
import itertools
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

#============================================================
# From the early labs...
#============================================================

def keys_geq_threshold (Dict, threshold):
    """
    (Generator) Given a dictionary, yields the keys whose values
    are at or above (greater than or equal to) a given threshold.
    """
    for key, value in Dict.items ():
        if value >= threshold:
            yield key

def alpha_chars (text):
    """
    (Generator) Yields each of the alphabetic characters in a string.
    """
    for letter in text:
        if letter.isalpha ():
            yield letter

def alpha_chars_pairs (text):
    """
    (Generator) Yields every one of the 4-choose-2 pairs of
    'positionally distinct' alphabetic characters in a string.
    
    That is, each position of the string is regarded as distinct,
    but the pair of characters coming from positions (i, j),
    where i != j, are considered the "same" as the paired
    positions (j, i). Non-alphabetic characters should be
    ignored.
    
    For instance, `alpha_chars_pairs ("te3x_t")` should produce
    has just 4 positionally distinct characters, so this routine
    should return the 4 choose 2 == 6 pairs:
      ('t', 'e')    <-- from positions (0, 1)
      ('t', 'x')    <-- from positions (0, 3)
      ('t', 't')    <-- from positions (0, 5)
      ('e', 'x')    <-- from positions (1, 3)
      ('e', 't')    <-- from positions (1, 5)
      ('x', 't')    <-- from positions (3, 5)
    """
    alpha_text = list (alpha_chars (text))
    return itertools.combinations (alpha_text)


from collections import defaultdict

def sparse_vector (base_type=float):
    return defaultdict (base_type)

def print_sparse_vector (x):
    for key, value in x.items ():
        print ("%s: %d" % (key, value))

def sparse_matrix (base_type=float):
    """
    Returns an empty sparse matrix that can hold integer counts
    of pairs of elements.
    """
    return defaultdict (lambda: sparse_vector (base_type))

def print_sparse_matrix (x):
    for i, row_i in x.items ():
        for j, value in row_i.items ():
            print ("[%s, %s]: %d" % (i, j, value))


def dense_vector (n, init_val=0.0):
    """
    [Lab 14] Returns a dense vector of length `n`, with all
    entries set to `init_val`.
    """
    return [init_val] * n

def spmv (n, A, x):
    """
    [Lab 14] Returns a dense vector y of length n, where
    y = A*x.
    """
    y = dense_vector (n)
    for (i, A_i) in A.items ():
        s = 0
        for (j, a_ij) in A_i.items ():
            s += a_ij * x[j]
        y[i] = s
    return y

import math

def vec_scale (x, alpha):
    """[Lab 14] Scales the vector x by a constant alpha."""
    return [x_i*alpha for x_i in x]

def vec_add_scalar (x, c):
    """[Lab 14] Adds the scalar value c to every element of x."""
    return [x_i+c for x_i in x]

def vec_sub (x, y):
    """[Lab 14] Returns x - y"""
    return [x_i - y_i for (x_i, y_i) in zip (x, y)]

def vec_2norm (x):
    """[Lab 14] Returns ||x||_2"""
    return math.sqrt (sum ([x_i**2 for x_i in x]))


import pandas as pd
import sys

def pandas2sqlite (df_reader, sql_writer, table_name, capitalize=False):
    """
    Given a text file reader for a Pandas data frame, creates an SQLite
    table. Returns the number of rows read.
    """
    index_start = 0
    for df in df_reader:
        if capitalize:
            df.columns = [x.capitalize () for x in df.columns.values]
        action = 'replace' if (index_start == 1) else 'append'
        df.to_sql (table_name, sql_writer, if_exists=action)
        index_start += len (df)
        
        print ("(Processed %d records.)" % index_start)
        sys.stdout.flush ()
    return index_start

def peek_table (db, name):
    """
    [Lab 14] Given a database connection (`db`), prints both the number of
    records in the table as well as its first few entries.
    """
    count = '''SELECT COUNT (*) FROM {table}'''.format (table=name)
    display (pandas.read_sql_query (count, db))
    peek = '''SELECT * FROM {table} LIMIT 5'''.format (table=name)
    display (pandas.read_sql_query (peek, db))


# [Lab 21] Floating-point utilities
# See also: https://docs.python.org/2/tutorial/floatingpoint.html
import re
from decimal import Decimal
import numpy as np

RE_FLOAT_HEX_PARTS = re.compile (r'''^(?P<sign>-)?0x[01]\.(?P<mantissa>[0-9a-f]+)p(?P<signexp>[+-])(?P<exp>\d+)''')

def float_to_bin (x):
    """Given a `float`, returns its binary form as a string."""
    assert type (x) is float
    s_hex = float.hex (x)
    hex_parts = RE_FLOAT_HEX_PARTS.match (s_hex)
    assert hex_parts
    
    s = hex_parts.group ('sign')
    m = hex_parts.group ('mantissa')
    se = hex_parts.group ('signexp')
    e = hex_parts.group ('exp')
    
    # Mantissa, including sign bit
    # See also: http://stackoverflow.com/questions/1425493/convert-hex-to-binary
    s_bin = '['
    if s:
        s_bin += s
    if (int (m, 16) == 0) and (int (e) == 0):
        s_bin += "0"
    else:
        s_bin += "1"
    s_bin += \
        "." \
        + bin (int (m, 16))[2:].zfill (4 * len (m)) \
        + "]_{2}"
    
    # Sign of exponent
    s_bin += "e" + se
    
    # Exponent
    s_bin += e

    return s_bin

# Copied here from Lab 21, for use starting in Lab 22 and beyond
def print_float_bin (x, prefix="", ret=False):
    s = ("%s: %s\n%s  %s" % (prefix,
                             Decimal (x),
                             ' ' * len (prefix),
                             float_to_bin (x))) 
    print (s)
    if ret:
        return s

EPS_S = np.finfo (np.float32).eps
EPS_D = np.finfo (float).eps

# ======================================================================
# [Lab 24] Linear regression

def generate_model (d):
    """Returns a set of (random) d+1 linear model coefficients."""
    return np.random.rand (d+1, 1)

def generate_data (m, x, sigma=1.0/(2**0.5)):
    """
    Generates 'm' noisy observations for a linear model whose
    predictor (non-intercept) coefficients are given in 'x'.
    Decrease 'sigma' to decrease the amount of noise.
    """
    assert (type (x) is np.ndarray) and (x.ndim == 2) and (x.shape[1] == 1)
    n = len (x)
    A = np.random.rand (m, n)
    A[:, 0] = 1.0
    b = A.dot (x) + sigma*np.random.randn (m, 1)
    return (A, b)

def linreg_fit_lstsq (A, b):
    """
    Solves Ax=b by a linear least squares method.
    """
    result = np.linalg.lstsq (A, b)
    x = result[0]
    return x

def rel_diff (x, y, ord=2):
    """
    Computes ||x-y|| / ||y||. Uses 2-norm by default;
    override by setting 'ord'.
    """
    return np.linalg.norm (x - y, ord=ord) / np.linalg.norm (y, ord=ord)

# ======================================================================
# [Lab 25] Logistic regression

import plotly.plotly as py
from plotly.graph_objs import *

def assert_points_2d (points):
    """Checks the dimensions of a given point set."""
    assert type (points) is np.ndarray
    assert points.ndim == 2
    assert points.shape[1] == 3
    
def assert_labels (labels):
    """Checks the type of a given set of labels (must be integral)."""
    assert labels is not None
    assert (type (labels) is np.ndarray) or (type (labels) is list)

def extract_clusters (points, labels):
    """
    Given a list or array of labeled augmented points, this
    routine returns a pair of lists, (C[0:k], L[0:k]), where
    C[i] is an array of all points whose labels are L[i].
    """
    assert_points_2d (points)
    assert_labels (labels)

    id_label_pairs = list (enumerate (set (labels.flatten ())))
    labels_map = dict ([(v, i) for (i, v) in id_label_pairs])
    
    # Count how many points belong to each cluster
    counts = [0] * len (labels_map)
    for l in labels.flatten ():
        counts[labels_map[l]] += 1
        
    # Allocate space for each cluster
    clusters = [np.zeros ((k, 3)) for k in counts]
    
    # Separate the points by cluster
    counts = [0] * len (labels_map)
    for (x, l) in zip (points, labels.flatten ()):
        l_id = labels_map[l]
        k = counts[l_id]
        clusters[l_id][k, :] = x
        counts[l_id] += 1
        
    # Generate cluster labels
    cluster_labels = [None] * len (labels_map)
    for (l, i) in labels_map.items ():
        cluster_labels[i] = l
        
    return (clusters, cluster_labels)

def make_2d_scatter_traces (points, labels=None):
    """
    Given an augmented point set, possibly labeled,
    returns a list Plotly-compatible marker traces.
    """
    assert_points_2d (points)
    
    traces = []
    if labels is None:
        traces.append (Scatter (x=points[:, 1:2], y=points[:, 2:3], mode='markers'))
    else:
        assert_labels (labels)
        (clusters, cluster_labels) = extract_clusters (points, labels)
        for (c, l) in zip (clusters, cluster_labels):
            traces.append (Scatter (x=c[:, 1:2], y=c[:, 2:3],
                                    mode='markers',
                                    name="%s" % str (l)))
    return traces

def heaviside_int (Y):
    """Evaluates the heaviside function, but returns integer values."""
    return heaviside (Y).astype (dtype=int)

def assert_discriminant (theta, d=2):
    """
    Verifies that the given coefficients correspond to a
    d-dimensional linear discriminant ($\theta$).
    """
    assert len (theta) == (d+1)

def lin_discr (X, theta):
    return X.dot (theta)

def heaviside (Y):
    return 1.0*(Y > 0.0)
    
def gen_lin_discr_labels (points, theta, fun=heaviside_int):
    """
    Given a set of points and the coefficients of a linear
    discriminant, this function returns a set of labels for
    the points with respect to this discriminant.
    """
    assert_points_2d (points)
    assert_discriminant (theta)
    
    score = lin_discr (points, theta)
    labels = fun (score)
    return labels

def gen_lin_discr_trace (points, theta, name='Discriminant'):
    """
    Given a set of points and the coefficients of a linear
    discriminant, this function returns a set of Plotly
    traces that show how the points are classified as well
    as the location of the discriminant boundary.
    """
    assert_points_2d (points)
    assert_discriminant (theta)
    
    x1 = [min (points[:, 1]), max (points[:, 1])]
    m = -theta[1] / theta[2]
    b = -theta[0] / theta[2]
    x2 = [(b + m*x) for x in x1]
        
    return Scatter (x=x1, y=x2, mode='lines', name=name)

def np_row_vec (init_list):
    """Generates a Numpy-compatible row vector."""
    return np.array (init_list, order='F', ndmin=2)

def np_col_vec (init_list):
    """Generates a Numpy-compatible column vector."""
    return np_row_vec (init_list).T

def check_labels (points, labels, fun):
    """
    Given a set of points and their labels, determines whether
    a given function produces matching labels.
    """
    your_labels = fun (points)
    return (labels == your_labels)

def logistic (Y):
    return 1.0 / (1.0 + np.exp (-Y))

#============================================================
# Adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
#============================================================

import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros
import urllib
import gzip

def download_mnist (data='training'):
    """
    Downloads gzip'd MNIST image + label data files into temporary local files,
    returning their filenames as a pair.
    """
    assert data in ['training', 'testing']
    
    if data == 'training':
        images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    else:
        images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        
    (images_fn_gz, _) = urllib.urlretrieve ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    (labels_fn_gz, _) = urllib.urlretrieve ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    return (images_fn_gz, labels_fn_gz)

def load_mnist (images_fn_gz, labels_fn_gz, digits=None, path=None, asbytes=False, selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. See: download_mnist()

    Parameters
    ----------
    images_fn_gz, labels_fn_gz : str
        Gzip'd filenames corresponding to MNIST-formatted files containing
        handwritten images and labels, respectively.
    dataset : str 
        Either "training" or "testing", depending on which dataset you want to
        load. 
    digits : list 
        Integer list of digits to load. The entire database is loaded if set to
        ``None``. Default is ``None``.
    path : str 
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST``. The data can
        be downloaded from http://yann.lecun.com/exdb/mnist/.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to
        ``numpy.float64`` in [0.0, 1.0].
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(0, 20, 2)``, which would load every other digit
        until--but not including--the twentieth.
    return_labels : bool
        Specify whether or not labels should be returned. This is also a speed
        performance if digits are not specified, since then the labels file
        does not need to be read at all.
    return_indicies : bool
        Specify whether or not to return the MNIST indices that were fetched.
        This is valuable only if digits is specified, because in that case it
        can be valuable to know how far
        in the database it reached.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. If neither labels nor inices are returned, then this is returned directly, and not inside a 1-sized tuple.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned only if ``return_labels`` is `True`, which is default.
    indices : ndarray
        The indices in the database that were returned.

    Examples
    --------
    This will load all images and labels from the training set:

    >>> im_gz, lab_gz = download_mnist ('training') # doctest: +SKIP
    >>> images, labels = load_mnist (im_gz, lab_gz) # doctest: +SKIP

    Load 100 sevens from the testing set:    

    >>> sevens = load_mnist (im_gz, lab_gz, digits=[7], selection=slice(0, 100), return_labels=False) # doctest: +SKIP

    """

    # We can skip the labels file only if digits aren't specified and labels aren't asked for
    if return_labels or digits is not None:
        flbl = gzip.open (labels_fn_gz, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = gzip.open(images_fn_gz, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection] 
    N = len(indices)

    images = zeros((N, rows, cols), dtype=uint8)

    if return_labels:
        labels = zeros((N), dtype=int8)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)
    if len(ret) == 1:
        return ret[0] # Don't return a tuple of one
    else:
        return ret

#============================================================
# [Labs 29 & 30] PCA + SVD
#============================================================

def peek_Sigma (Sigma, ret_df=False):
    k = len (Sigma)
    df_Sigma = pd.DataFrame (np.arange (len (Sigma)), columns=['i'])
    df_Sigma['sigma_i'] = Sigma
    Sigma_sq = np.power (Sigma, 2)
    Err_sq = np.sum (Sigma_sq) - np.cumsum (Sigma_sq)
    Err_sq[Err_sq < 0] = 0
    Err = np.sqrt (Err_sq)
    Relerr = Err / (Sigma[0] + Err[0])
    df_Sigma['sigma_i^2'] = Sigma_sq
    df_Sigma['err_i^2'] = Err_sq
    df_Sigma['err_i'] = Err
    df_Sigma['relerr_i'] = Relerr
    print ("Singular values:")
    display (df_Sigma.head ())
    print ("  ...")
    display (df_Sigma.tail ())
    
    f, ax = plt.subplots (figsize=(7, 7))
    ax.set (yscale="log")
    sns.regplot ("i", "sigma_i", df_Sigma, ax=ax, fit_reg=False)
    if ret_df:
        return df_Sigma

def rgb2gray (rgb):
    return np.dot (rgb[...,:3], [0.299, 0.587, 0.144])

def imshow_gray (im):
    plt.imshow (im, interpolation='nearest', cmap=plt.get_cmap ('gray'))

#============================================================
if __name__ == "__main__":
    print (__doc__)
#============================================================

# eof
