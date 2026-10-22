########### Code for calculating 2D dispersion images quickly.
########### Author: Nikhil Punithan, Colorado School of Mines, Fall 2026, adapted from Eileen R. Martin, Stanford University, Dec. 2015
########### Available at github.com/Nik-P2/FastDispersionImages

import numpy as np
import scipy
import dascore as dc

def sigma_calc(patch, phase_vels, approx_resolution = None, approx_freqs=None):
    """
    calculates the sigma factor defined in page 85 of Dr. Eileen Martin's PhD dissertation
    Parameters
    ----------
    patch: Dascore Patch 
        Raw or preprocessed DAS data

    phase_vels: 1D array
        array containing all the phase velocities to construct a dispersion image for

    approx_resolution: Float
        desired frequency resolution
        
    approx_freqs: 1D array: tuple
        array containing the start and end frequency to construct a dispersion image for
    
    Returns
    -------
    sigma: 2D array with dimensions of len(phase_vels)*len(freqs)
    """

    # slowness
    p = 1.0/phase_vels # convert from velocity to slowness
    
    # reshape patch to ensure we access the proper axis
    re_patch = patch.transpose("time", "distance")

    # extract data and coordinates
    dat = re_patch.data
    t_coord = re_patch.coords.get_array("time")
    d_coord = re_patch.coords.get_array("distance")

    dt = float((t_coord[1]-t_coord[0])/np.timedelta64(1, "s"))

    nt, n_dist = dat.shape

    if dt <=0:
        raise ValueError("Time sampling interval must be positive")

    # determine nyquuist frequency to check inputs
    fs = 1.0/dt
    nyquist = fs/2.0

    # determine fft length from approx_resolution
    if approx_resolution is None:
        n_fft = nt
    else:
        if approx_resolution <= 0:
            raise ValueError("approx_resolution must be positive")
        n_fft = max(1, int(round(fs/approx_resolution)))

    # determine bounds for frequency calculation
    if approx_freqs is None:
        f_min = 0.0
        f_max = nyquist
    else:
        f_min = approx_freqs[0]
        f_max = approx_freqs[-1]

        if f_min < 0 or f_max <= f_min:
            raise ValueError("approx_freqs must contain an increasing, nonnegative frequency range")
        if f_max > nyquist:
            raise ValueError("Maximum frequency cannot exceed Nyquist: "+ str(nyquist))

    # calculate fft along time axis
    spec_data = scipy.fft.fft(dat, n=n_fft, axis = 0).astype(np.complex64, copy=False)

    # calculate fft coords
    freq_arr = scipy.fft.fftfreq(n_fft, d = dt)

    pos_mask = ((freq_arr >= f_min) & (freq_arr <= f_max) & (freq_arr > 0)) # mask to select positive frequencies
    neg_mask = ((freq_arr >= -f_min) & (freq_arr <= -f_max) & (freq_arr < 0)) # mask to select negative frequencies

    pos_freq = freq_arr[pos_mask] # positve frequency vals
    neg_freq = freq_arr[neg_mask] # negative frequency vals

    # put positive and negative frequencies together
    pa_freq = np.concatenate((pos_freq, neg_freq))

    # isolate positive and negative fourier transforms
    pos_spec = spec_data[pos_mask, :]
    neg_spec = spec_data[neg_mask, :]

    # combine fourier transforms together
    pos_neg_spec = np.concatenate((pos_spec, neg_spec), axis = 0)

    spec_patch = dc.Patch(data = pos_neg_spec, coords = {"ft_time" : pa_freq, "distance" : d_coord}, dims = ("ft_time", "distance"), attrs = re_patch.attrs)

    # calculate sigma over chunked phase velocities to manage memory
    sigma = np.empty((len(phase_vels), len(pa_freq)), dtype=np.complex64) #create empty array for sigma

    chunk = 32 # chunk size is 32

    for i0 in range(0, len(phase_vels), chunk): # loop through phasevels in chunks
        i1 = min(i0 + chunk, len(phase_vels)) # determine end index for a chunk
        phaseShiftMat = np.exp(2j*np.pi*p[i0:i1, None, None]*pa_freq[None,:, None]*d_coord[None, None, :]) # calculate phase shift for receivers
        sigma[i0:i1] = np.einsum("vfd,fd->vf", phaseShiftMat, pos_neg_spec, optimize=True) # calculate a chunk of sigma

    return sigma, spec_patch, pa_freq

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
    dsp_sym = dsp[:, :len(pos_freq)] + (dsp[:, len(neg_freq):])

    # convert dsp into a dascore Patch, similar to the dispersion_phaseshift function
    attrs = dict(category = "DAS")
    coords = dict(velocity=phase_vels, frequency = pos_freq)
    dsp_patch = dc.Patch(data = dsp_sym, coords = coords, attrs = attrs, dims=["velocity", "frequency"])

    return dsp_patch.set_units(velocity="m/s", frequency="Hz")

def dsp_stack_calc(patch, phase_vels, r_primes, approx_resolution = None, approx_freqs=None):
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
    sigma_factor, sub_spec, pa_freq = sigma_calc(patch, phase_vels, approx_resolution=approx_resolution, approx_freqs=approx_freqs)

    # extract array of al distance positions from patch
    dist_arr = patch.coords.get_array("distance")

    dsp_stack_list = [] # create empty list to hold dipsersion images per virtual source

    for i in range(len(r_primes_list)): # iterates through r_primes (converts r_primes to a 1 length array if input is a single int)
        source_index = np.argmin(np.abs(dist_arr - r_primes_list[i]))# convert from r_primes to an index location in the data, via minimum difference
        dsp_stack_list.append(dsp_calc(sub_spec.data[:, source_index], sub_spec.coords.get_array("distance")[source_index], phase_vels, pa_freq, sigma_factor))

    return dsp_stack_list