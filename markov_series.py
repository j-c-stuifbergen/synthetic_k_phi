# -*- coding: utf-8 -*-
"""
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
Acknowledgements:
The function "create_approximately_structured_single_variable" was originally written by Co Stuifbergen in JavaScript and translated to Python by HARBR.
"""

"""
from scipy.constants import g
from scipy.linalg import eigh, cholesky
"""
import tkinter
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm
from scipy import stats
from sklearn.cluster import KMeans

###############################################################################
# if reproducible results are needed: use a specific seed number, otherwise 
# comment out
random.seed(1000) 
#

# min/max for probability map

lbound = 0
ubound = 5

# resolution of the probability distributions
def create_markov_series(n=100, nbins = 100, opt_plot_prob_map=True, opt_plot_dummy_var=True):
    '''This function creates an array with a single dummy variable that is 
    "approximately sorted" (i.e. more structured than random) to resemble true 
    geology better. The "approximate sorting" is obtained by picking a random 
    value based on a probability map defined by two normal distributions.
    
    The only arguments required are the variables that define a sum of gaussian
    distributions on the probability map (function "probability_definition") and 
    the length of the dummy data array (obtained automatically).
    
    -- The optional argument opt_plot_prob_map (default=True, otherwise False)
       shows/not shows a visual representation of the probability map
    -- The optional argument opt_plot_dummy_var (default=True, otherwise False)
       shows/not shows the creates (approximately sorted) dummy variable
     
    This function was written by Co Stuifbergen in JavaScript, translated to
    Python by HARBR (30th December 2019)'''
     
    def probability_definition(x):
        '''Defines the "map" with probability density distributions
        x, y are the x and y data
        The current definition makes a linear combination of 2 Gaussian 
        distributions (the centers of which are defined by the variable
        peaks)
        '''

        # for probability map: positions, amplitude and width of peaks:
        # syntax: peak_n = tuple(xoffset, yoffset, amplitude, peakwidth)
        # peakwidth needs to be > 0. Larger number will give a wider peak
        peak_1 = (0.75, 0.75, 2.00, 0.2)
        peak_2 = (2.25, 2.25, 1.00, 0.3)
        #peak_3 = (2.00, 2.25, 1.00, 0.3)
        peaks = [peak_1, peak_2] #, peak_3]

        def prob_out(y):
            result = 0
            for p in peaks:
                xoffset = p[0]
                yoffset = p[1]
                ampl = p[2]
                peakwidth = p[3]
                r2 = math.pow(x-xoffset,2) + math.pow(y-yoffset,2)
                result += ampl*math.exp(-r2/peakwidth)
            return(result)
        return prob_out    
    
    def bins(start,end,n,yi):
        '''Small function to set the number of bins at the "cross-section"
        @yi: the cumulative distribution curve is divided into n equal-sized bins.
        Note that the result is not normalised, i.e. y[n-1] is not 1.0 '''
        probaSelected = probability_definition(yi)
        width = (end-start)/n
        y = []
        y.append(probaSelected(start+0.5*width))
        for i in range(1,n):
            y.append(y[i-1] + probaSelected((i+0.5)*width))
        return(y)   
    
    
    def scaled_random(mymin,mymax):
        '''Small function to scale the random value between lbound and ubound'''
        return(mymin + random.random()*(mymax-mymin))
     
    
    myresults = [] # create empty list to store results
    r = scaled_random(lbound,ubound)

    for i in range(n):
        # z = the binned cumulative probabilty at the specific value
        z = bins(lbound,ubound,nbins,r)
        # small function to scale random value
        r = scaled_random(0,z[nbins-1])
    
        if (r<z[0]):
            myresults.append(lbound+(ubound-lbound)/nbins*(r/z[0]))
        else:
            j = 1
            while(z[j]<r and j<nbins):
                j += 1
            myresults.append(lbound+(j+(r-z[j-1])/(z[j]-z[j-1]))*(ubound-lbound)/nbins)
        r = myresults[i]

    # This part is only to visualize the probability map - not necessary for 
    # the program to function properly 
    if opt_plot_prob_map == True:
        # Create arrays for the probability map
        # N.B. the number of points in the isa reduced by a factor 10 to speed up
        # (it's for visualization only anyway)
        nX = 50
        nY = 50
        xCoords = np.linspace(lbound,ubound,nX) 
        yCoords = np.linspace(lbound,ubound,nY)

        # create a meshgrid
        xxCoords,yyCoords = np.meshgrid(xCoords,yCoords,sparse=True)
        probDefiner = np.zeros((len(xCoords),len(yCoords)))
        # normed = np.zeros((len(xCoords),len(yCoords)))
        x = np.zeros(len(xCoords))

        figB = plt.figure(figsize=(6,6))
        bx = figB.gca(projection='3d')
        bx.set_title('probabilities');
        # create a mapping to the interval [0,1].
        # I don't want to use the extreme colors
        cn = colors.Normalize(-nX*0.3, len(xCoords)*1.5)
        # plt.set_cmap("inferno") # there should be a way to set the color map 

        # calculate "z"-value
        for i in range(len(xCoords)):
            probaSelected = probability_definition(xCoords[i])
            sum = 0
            zval = np.zeros(len(yCoords))
            for j in range(len(yCoords)):
                probDefiner[j,i]=probaSelected(yCoords[j])
                sum += probDefiner[j,i]
            for j in range(len(yCoords)):
                zval[j]   = probDefiner[j,i]/sum
                # normed[j,i] = zval[j]
            x= [xCoords[i]] * nY
            bx.plot(x,yCoords,zval, color = plt.cm.jet(cn(i)))
        #    ax.plot3D(x, yCoords, zval,  label='probabilities')
            
        # draw the probability density map
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        # ax = plt.axes(projection='3d')
        ax.plot_surface(xxCoords, yyCoords, probDefiner, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        ax.set_title('defining functions');

        '''
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        ax.plot_surface(xxCoords, yyCoords, normed, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        ax.set_title('probabilities');
        '''
        
    # convert results to pandas dataframe for easy plotting agains index

    if opt_plot_dummy_var : 
        # show the created dummy variable
        fig = plt.figure(figsize=(4,8))
        plt.plot(myresults,range(len(myresults)),'g-')
        plt.gca().invert_yaxis()
        
    if opt_plot_prob_map or opt_plot_dummy_var : 
        plt.show()

    return(myresults)



