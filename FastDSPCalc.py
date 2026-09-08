########### Code for calculating 2D dispersion images quickly.
########### Author: Nikhil Punithan, Colorado School of Mines, Fall 2026, adapted from Eileen R. Martin, Stanford University, Dec. 2015
########### Available at github.com/Nik-P2/FastDispersionImages

import numpy as np
import scipy
import dascore as dc

def sigma_calc(patch, phase_vels, freqs):
    """
    calculates the sigma factor defined in page 85 of Dr. Eileen Martin's PhD dissertation
    Parameters
    ----------
    patch: Dascore Patch 
        Raw or preprocessed DAS data

    phase_vels: 1D array
        array containing all the phase velocities to construct a dispersion image for

    freqs: 1D array
        array containing all the frequencies to construct a dispersion image for
    
    Returns
    -------
    sigma: 2D array with dimensions of len(phase_vels)*len(freqs)
    """

    # define the dimension of sigma
    nVel = len(phase_vels)
    nFreq = len(freqs)

    # initialize sigma as a 2D array filled with zeros
    sigma = np.zeros((nVel, nFreq))

    # don't need to calc frequencies of interest since that is already a user input

    # define the phase shift matrix
    p = 1.0/phase_vels # convert from velocity to slowness
    phaseShiftMat = np.exp(2*np.pi*1j*np.outer(p, freqs)) #phase shift matrix with dimensions the same as sigma and the final dispersion image

    # reshape patch to ensure we access the trace axis
    re_patch = patch.transpose("distance", "time")

    # calculate the spectrum of each trace
    spec = re_patch.dft(dim="time")

    # window to the specific frequencies of interest
    spec_sel = spec.select(ft_time = (freqs[0], freqs[-1]))

    # change data format to float to avoid data format error
    sub_spec = spec_sel.update(data = spec_sel.data.astype(np.float64))

    # apply the phase shift by looping through each individual trace
    for i in range(len(sub_spec.data)):
        # isolate a single traces spectrum 
        spec_trace = sub_spec.data[i,:]

        sigma = sigma + np.tile(spec_trace, (nVel,1)) * np.power(phaseShiftMat, sub_spec.coords.get_array("distance")[i])

    return sigma, sub_spec

def dsp_calc(sub_spec_r, trace_pos, phase_vels, freqs, sigma):
    """
    calculates a dispersion image for a singular virtual source using 
    Parameters
    ----------
    sub_spec_r: 1D array
        virtual source trace spectrum, should already be windowed to desired frequency range
    
    trace_pos: Float
        position along fiber in meters of the virtual source

    phase_vels: 1D array
        array containing all the phase velocities to construct a dispersion image for
    
    freqs: 1D array
        array containing all the frequencies to construct a dispersion image for
    
    sigma: 2D array
        array containing the precomputed sigma values for every trace in the patch   
    Returns
    -------
    dsp: 2D array
        dispersion image of the data
    """

    # define dimensions for spectrum
    nVel = len(phase_vels)
    nFreq = len(freqs)

    # define the phase shift matrix
    p = 1.0/phase_vels # convert from velocity to slowness
    phaseShiftMat = np.exp(-2*np.pi*1j*np.outer(p, freqs)) #phase shift matrix with dimensions the same as sigma and the final dispersion image

    # dispersion calc by multiplying the data spectrum by the phase shifts
    dsp = np.abs(np.tile(np.conj(sub_spec_r),(nVel,1))*np.power(phaseShiftMat,trace_pos)*sigma)

    return dsp

def dsp_stack_calc(patch, phase_vels, freqs, r_primes, supergather):
    """
    calculates dispersion images for all possible virtual sources in a patch
    Parameters
    ----------
    patch: Dascore Patch 
        Raw or preprocessed DAS data
    
    phase_vels: 1D array
        array containing all the phase velocities to construct a dispersion image for
    
    freqs: 1D array
        array containing all the frequencies to construct a dispersion image for

    r_primes: 1D array or an Int
        an Int or array that holds the virtual source locations in meters

    supergather: Boolean
        if True, stack all dispersions across all virtual sources together
        if False, place each separate dispersion in an array, 

    Returns
    -------
    dsp_stack: 2D array
        stacked dispersion image over all virtual sources

    dsp_stack_list: list of 2D arrays
            dispersion images for each virtual source in an order list
    """

    # calculate the sigma factor for this patch
    sigma_factor, sub_spec = sigma_calc(patch, phase_vels, freqs)

    # extract array of al distance positions from patch
    dist_arr = patch.coords.get_array('distance')

    # for each virtual source do a dispersion calculation using sigma_factor
    if supergather == True and not isinstance(r_primes, int):# if multiple virtual sources are input and we want a supergather
        dsp_stack = np.zeros((len(phase_vels), len(freqs)))# set up dsp_stack as an empty array

        for i in range(len(r_primes)):# loop through r_primes
            source_index = np.argmin(np.abs(dist_arr - r_primes[i]))# convert from r_primes to an index location in the data, via minimum difference
            dsp_stack += dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, freqs, sigma_factor)

        return dsp_stack

    elif isinstance(r_primes, int):# if only one virtual source is input
        dsp_stack = np.zeros((len(phase_vels), len(freqs)))# set up dsp_stack as an empty array
        source_index = np.argmin(np.abs(dist_arr - r_primes))# convert from r_primes to an index location in the data, via minimum difference
        dsp_stack += dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, freqs, sigma_factor)

        return dsp_stack

    else:# if we want multiple virtual sources that are separate
        dsp_stack_list = []# set up dsp_stack_list as an empty list

        for i in range(len(r_primes)):
            source_index = np.argmin(np.abs(dist_arr - r_primes[i]))# convert from r_primes to an index location in the data, via minimum difference
            dsp_stack_list.append(dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, freqs, sigma_factor))

        return dsp_stack_list


'''
if im calling the above function through a function that gets looped through time, and sigma already calculates the spectrum
then maybe I also include a return for the sub_spec patch so that the loop over virtual sources also has access to that data since it shouldn't
be particularly large?


include an optional input for dsp_stack_calc where user inputs an array of virtual source locations and we only iterate through those?
'''

'''
recalculating the spectrum might be necessary if the time windows we work with are too large 
'''

'''
time optimizations
-use broadcasting to get rid of loop in sigma_calc
-when calculating phaseShiftMat use broadcasting to get rid of the expensive np.outer calc
-precompute phaseShiftMat so it's not done every single call of dsp_calc
-in dsp_calc use broadcasting to get around the use of np.tile
-
'''