#!/usr/bin/env python3
"""
Module: mnist

Some utility functions created for Georgia Tech's CSE 6040: Computing for Data Analysis.
These functions pertain to downloading and manipulating the MNIST handwriting digit
recognition dataset.

Adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
"""

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
        
    (images_fn_gz, _) = urllib.request.urlretrieve ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    (labels_fn_gz, _) = urllib.request.urlretrieve ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
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
if __name__ == "__main__":
    print (__doc__)
#============================================================

# eof
