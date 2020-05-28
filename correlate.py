# -*- coding: utf-8 -*-
"""
This program produces a synthetic k-phi dataset (vs depth) from user-specified
Petrophysical Rock Types (PRT).
The result is a incidental lithology column based on the PRT entered by the
user with accompanying porosity/permeability data - all depth indexed.

The code works such that a random (but more structured than complete random)
ordering of the PRTs is generated: a synthetic lithology column. Due to this 
incidental structuring, this column resembles more "true geology" than what 
would be generated from a complete random state.
Once the "stacking" of PRTs is in place, the random variables are transformed
according to user-specified statistics for each of the PRTs:
    -- 

The user needs to specify the following (in the "USER SETUP" section below):
    -- litholib: a dictionary containing PRTs + characteristics
       {#, ('name prt', 'plot color', 'hash-symbol', 'correlation metric',
            'X5 (5th percentile) phi distribution, X95 phi, X5 k, X95 k)}
        example:   
        litholib = {0:('shale', 'brown',  '--',0.92, 0.07, 0.14, 0.008, 1.5),
                    3:('sst',   'yellow', '.', 0.97, 0.14, 0.28, 10,   5000)}
    -- Sampling rate (sr, in /m)
    -- Top of interval (top_int, m)
    -- Length of interval (length_of_interval, m)
    -- show_intermediate_plots: True/False. This option will show/hide the 
       plots that are created at intermediate stages. These could be useful
       for QC
    -- method defines the method how the correlated variables are created. The
       choices are "cholesky" or "eigh". First, uncorrelated, random, normally 
       distributed variables are generated, which are subsequently multiplied
       by a matrix such that CC^T = R. R is the desired covariance matrix. C is 
       created by either Cholesky decomposition of R (method="cholesky"), or 
       from the eigenvalues and eigenvectors of R (method="eigh").
       Is seems that "Cholesky" generally results in a little higher correlation
       coefficients on the final data tha "eigenvectors and eigenvalues"

It also is possible to change the probability definition map also. This is 
somewhat trial and error, but the program works best when the peaks lie on a 
diagonal. Adjust the variables "peak_1" and "peak_2" to change the probability
distribution map. The variables "peak_n" are tuples containing the following 
elements:
    
    peak_n = (xoffset, yoffset, amplitude, peakwidth)
    -- xoffset, yoffset define the position of the peak on the probability 
       distribution map
    -- amplitude defines the height of the peak (larger number increases
       probability)
    -- peakwidth define the width of the distribution. The number needs to be 
       greater than 0. Larger numbers increase the width (variance) of the
       distribution

Unless a seed number is entered, the code will produce a new, random dateset
each time it is run. If repeatability is required, fix the random state by 
entering a (any) seed-number.

Known issues:
    -- The lithology/PRT column isn't drawn entirely correct. Especially when
       more than 2 PRTs are used, or when the column is relatively short, the
       colors visible do not seem to correspond to the occurrence of each of
       the PRTs (see percentages top of plot). The data (stored in the pandas 
       DataFrame) are correct though (i.e. it is a graphics or plotting issue)
       
    -- The correlation metrics in the "litholib"/PRT dictionary need to be 
       quite high, even for PRTs that will ultimately be poorly correlated.
       This is unfortunate, as it is somewhat trial and error to get a "good-
       looking" dataset.
       
       
Created December 2019
@author: HARBR (Harry Brandsen)

Acknowledgements:
The function "create_approximately_structured_single_variable" was written
by Co Stuifbergen in JavaScript and translated to Python by HARBR.
"""

import numpy as np

from scipy.linalg import eigh, cholesky
from scipy.stats import norm

###############################################################################
method = 'cholesky'             # method needs to be either 'cholesky' or 'eigh'
###############################################################################

def double_dummy_data(x):
    '''This small function creates a 2 by n array to later use to generate 
    correlated data. The upper row in this array is replaced by the earlier-
    created "approximately sorted" data (the "x" argument). Depending on the
    method chosen ("cholesky" or "eigh"), the upper resp. lower row needs to
    be replaced (this is the row that remains the same during the matrix 
    transformation)'''
    # generate random data
    x2 = norm.rvs(size=len(x))

    # append to the given data
    xy = np.vstack((x,x2))
    
    return(xy)



def correlate_doubled_dummy_data(xy,method,corr):
    '''Creates aset of two correlated variables. The user can indicate:
    -- xy is the 2 by n array of dummy data in which the upper row is the
       "approximately sorted" dummy data
    -- the method = "cholesky" or "eigh",
    -- the correlation coefficient r
    
    See https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
    for details.'''
            
    # The desired covariance matrix.
    if corr > 1 or corr < -1:
        print('The correlation coefficient must be between -1 and 1 incl.')
        return
    
    r = np.array([
            [  1, corr],
            [ corr,  1]])
    
    # Generate samples from three independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
#    x = norm.rvs(size=(2, n))
    
    # We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
    # the Cholesky decomposition, or the we can construct `c` from the
    # eigenvectors and eigenvalues.
    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    elif method == 'eigh':
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = eigh(r)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
        # exchange columns, so the result will be similar to
        # the results of the Cholesky decomposition
        p = np.array([
            [  0, 1],
            [ 1,0]])
        c= np.dot(c,p)
    else:
        raise Exception("unknown method entered for matrix decomposition, choose 'eigh' or 'cholesky'")
    
    # Convert the data to correlated random variables. 
    return(np.dot(c, xy))

