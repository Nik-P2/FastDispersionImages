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
    
    # reshape patch to ensure we access the proper axis
    re_patch = patch.transpose("time", "distance")

    # calculate the spectrum of each trace
    spec = re_patch.dft(dim="time")

    # window to the specific frequencies of interest
    pos_spec = spec.select(ft_time = (freqs[0], freqs[-1])) # positive spec
    neg_spec = spec.select(ft_time = (-freqs[-1], -freqs[-0])) # negative spec

    # get positive and negative frequency arrays
    pos_freq = pos_spec.coords.get_array("ft_time") # coords for positive frequency metadata
    neg_freq = neg_spec.coords.get_array("ft_time") # coords for negative frequency metadata

    # make freqency and distance arrays
    pa_freq = np.hstack((pos_freq, neg_freq)) # this should be something like 1, 2, 3, 4, 5, -5, -4, -3, -2, -1 
    pa_dist = spec.coords.get_array("distance")

    # calculate the phase shift matrix, has shape of (nVel, nFreq, nDistance)
    phaseShiftMat = np.exp(2j * np.pi * p[:, None, None] * pa_freq[None, :, None] * pa_dist[None, None, :])

    # has shape of (nFreq, nDistance)
    # make concatenated spectrum data with shape of (2*nFreq, nDistance)
    tempspool = dc.spool([pos_spec, neg_spec])
    freq_concat = tempspool.concatenate(ft_time=None)

    pos_neg_spec = freq_concat[0].data

    # has shape of (nVel, nFreq)
    sigma = np.sum(phaseShiftMat * pos_neg_spec[None, :, :], axis = 2)

    return sigma, freq_concat[0], pa_freq

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
    
    pa_freq: 1D array
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
    dsp = np.conj(sub_spec_r)[None, :] * phaseShiftMat * sigma

    # symmetrize the positive and negative frequencies in the dispersion image
    pos_freq = pa_freq[pa_freq > 0]
    neg_freq = pa_freq[pa_freq < 0]
    dsp_sym = dsp[:, :len(pos_freq)] + np.fliplr(dsp[:, len(neg_freq):])

    # convert dsp into a dascore Patch, similar to the dispersion_phaseshift function
    attrs = dict(category = "DAS")
    coords = dict(velocity=phase_vels, frequency = pos_freq)
    dsp_patch = dc.Patch(data = dsp_sym, coords = coords, attrs = attrs, dims=["velocity", "frequency"])

    return dsp_patch.set_units(velocity="m/s", frequency="Hz")

def dsp_stack_calc(patch, phase_vels, freqs, r_primes):
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

    Returns
    -------
    dsp_stack_list: list of 2D arrays
            dispersion images for each virtual source in an order list

    !! important note !!
    --------------------
    The returned dispersion image(s) will be complex, the absolute value should be taken after all time windows are stacked through,
    otherwise the final dispersion image will be similar to a supergather regardless of virtual source
    """
    # convert r_primes into a list if it isn't already
    r_primes_list = np.atleast_1d(r_primes)

    # calculate the sigma factor for this patch
    sigma_factor, sub_spec, pa_freq = sigma_calc(patch, phase_vels, freqs)

    # extract array of al distance positions from patch
    dist_arr = patch.coords.get_array("distance")

    dsp_stack_list = [] # create empty list to hold dipsersion images per virtual source

    for i in range(len(r_primes_list)): # iterates through r_primes (converts r_primes to a 1 length array if input is a single int)
        source_index = np.argmin(np.abs(dist_arr - r_primes_list[i]))# convert from r_primes to an index location in the data, via minimum difference
        dsp_stack_list.append(dsp_calc(sub_spec.data[:, source_index], sub_spec.coords.get_array("distance")[source_index], phase_vels, pa_freq, sigma_factor))

    return dsp_stack_list