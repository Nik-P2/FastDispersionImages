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

    freqs: 1D array, tuple
        array containing the start and end frequency to construct a dispersion image for
    
    Returns
    -------
    sigma: 2D array with dimensions of len(phase_vels)*len(freqs)
    """

    # don't need to calc frequencies of interest since that is already a user input

    # slowness
    p = 1.0/phase_vels # convert from velocity to slowness
    
    # reshape patch to ensure we access the trace axis
    re_patch = patch.transpose("distance", "time")

    # calculate the spectrum of each trace
    spec = re_patch.dft(dim="time")

    # window to the specific frequencies of interest
    spec_sel = spec.select(ft_time = (freqs[0], freqs[-1]))

    pa_freq = spec_sel.coords.get_array("ft_time")
    pa_dist = spec_sel.coords.get_array("distance")

    # calculate the phase shift matrix, has shape of (nVel, nFreq, nDistance)
    phaseShiftMat = np.exp(2j * np.pi * p[:, None, None] * pa_freq[None, :, None] * pa_dist[None, None, :])

    # has shape of (nFreq, nDistance)
    spectrum = (spec_sel.data).T

    # has shape of (nVel, nFreq)
    sigma = np.sum(phaseShiftMat * spectrum[None, :, :], axis = 2)

    return sigma, spec_sel

def dsp_calc(sub_spec_r, trace_pos, phase_vels, pa_freq, sigma):
    """
    calculates a dispersion image for a singular virtual source using a precomputed sigma
    Parameters
    ----------
    sub_spec_r: 1D array
        virtual source trace spectrum, should already be windowed to desired frequency range
    
    trace_pos: Float
        position along fiber in meters of the virtual source

    phase_vels: 1D array
        array containing all the phase velocities to construct a dispersion image for
    
    pa_freqs: 1D array
        array containing all the frequencies to construct a dispersion image for
    
    sigma: 2D array
        array containing the precomputed sigma values for every trace in the patch   
    Returns
    -------
    dsp: 2D array
        dispersion image of the data
    """

    # define the phase shift matrix
    p = 1.0/phase_vels # convert from velocity to slowness

    phaseShiftMat = np.exp(-2j * np.pi * p[:, None] * pa_freq[None, :] * trace_pos) #phase shift matrix with dimensions the same as sigma and the final dispersion image

    # dispersion calc by multiplying the data spectrum by the phase shifts
    dsp = np.abs(np.conj(sub_spec_r)[None, :] * phaseShiftMat * sigma)

    # convert dsp into a dascore Patch, similar to the dispersion_phaseshift function
    attrs = dict(category = "DAS")
    coords = dict(velocity=phase_vels, frequency = pa_freq)
    dsp_patch = dc.Patch(data = dsp, coords = coords, attrs = attrs, dims=["velocity", "frequency"])

    return dsp_patch.set_units(velocity="m/s", frequency="Hz")

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
    dist_arr = patch.coords.get_array("distance")
    pa_freq = sub_spec.coords.get_array("ft_time")

    # for each virtual source do a dispersion calculation using sigma_factor
    if supergather == True and not isinstance(r_primes, int):# if multiple virtual sources are input and we want a supergather
        dsp_stack = 0# set up dsp_stack as an empty variable

        for i in range(len(r_primes)):# loop through r_primes
            source_index = np.argmin(np.abs(dist_arr - r_primes[i]))# convert from r_primes to an index location in the data, via minimum difference
            dsp_stack += dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, pa_freq, sigma_factor)

        return dsp_stack

    elif isinstance(r_primes, int):# if only one virtual source is input
        dsp_stack = 0# set up dsp_stack as an empty variable
        source_index = np.argmin(np.abs(dist_arr - r_primes))# convert from r_primes to an index location in the data, via minimum difference
        dsp_stack += dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, pa_freq, sigma_factor)

        return dsp_stack

    else:# if we want multiple virtual sources that are separate
        dsp_stack_list = []# set up dsp_stack_list as an empty list

        for i in range(len(r_primes)):
            source_index = np.argmin(np.abs(dist_arr - r_primes[i]))# convert from r_primes to an index location in the data, via minimum difference
            dsp_stack_list.append(dsp_calc(sub_spec.data[source_index,:], sub_spec.coords.get_array("distance")[source_index], phase_vels, pa_freq, sigma_factor))

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
-use broadcasting to get rid of loop in sigma_calc ***Done***
-when calculating phaseShiftMat use broadcasting to get rid of the expensive np.outer calc ***Done***
-precompute phaseShiftMat so it's not done every single call of dsp_calc
-in dsp_calc use broadcasting to get around the use of np.tile ***Done***
-
'''