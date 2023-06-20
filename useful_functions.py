# Miscellaneous useful functions written by Kenneth Ooi

import IPython
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import csv, librosa, librosa.display, os, pyaudio, seaborn, time, wave

__version__ = '20200615'

#%%================================================================== add_noise ========================================================================%%#

def add_noise(input_signal = np.sin(np.linspace(0,440,44100)*np.pi), input_noise = 2*np.random.rand(44100)-1, SNR = 0, normalization_mode = 'scale', verbose = False):
    '''
    Adds noise to a signal at a specified signal-to-noise ratio
    (SNR) expressed in dB. If the noisy signal has values outside
    the range [-1,1], its values are normalized according to the
    normalization mode specified.

    ========
     Inputs
    ========
    
    input_signal : np.array
        A single-channel signal (e.g. numpy.array) containing
        values between -1 and 1.
        Default is a one-second 440Hz sine tone sampled at
        44100Hz.

    input_noise : np.array
        A single-channel noise (e.g. numpy.array) containing
        values between -1 and 1.
        Default is a one-second uniform random noise track sampled
        at 44100Hz.
        
    SNR : float or np.Inf
        The signal-to-noise ratio, in dB, that the final noisy
        signal should have. If np.Inf, then exactly input_signal
        is returned. If np.NINF, then exactly input_noise is
        returned.
        
    normalization_mode : str
        Either 'scale' or 'clip'.
        If 'scale', then a gain will be applied to the entire
        noisy signal in order to bring the noisy signal to [-1,1],
        if it has values outside this range. If 'clip', then
        values outside of [-1,1] will be set to either -1 or 1,
        whichever is closer to that value.
        
    verbose : bool
        If true, prints out messages on what function is doing.
        If false, nothing is printed out.
 
    =========
     Outputs
    =========
    
    noisy_signal : np.array
        A single-channel noisy signal made from input_signal and
        input_noise, such that the signal-to-noise ratio of
        input_signal and input_noise is SNR.
    
    ======
     Idea
    ======
    
    Let SNR be the desired SNR of the signal wrt noise, in dB
    (i.e. SNR = 30 means that the signal power is 30dB greater
    than that of the noise power). In other words, we have to
    multiply the noise by some constant sqrt(K) to get SNR,
    since the signal and noise start off with arbitrary power.
    Recall that the power of a signal X is just E(X^2), so
    multiplying a signal by sqrt(K) will multiply its power by K.
    Now, let the signal and noise power be S and N respectively,
    then we will have

    10lg(S/(K*N)) = SNR
    -10lg(K*N/S) = SNR
    K = (S/N)*(10**(-SNR/10)),

    which is the calculation we perform here.
    
    ==============
     Dependencies
    ==============
    
    numpy (as np)
    
    ==========
     Examples
    ==========
    
    # Define two signals with equal power and attempt to combine them with 0 dB SNR.
    # The output signal should theoretically be x + y.
    >>> x = np.array([  0.5, 0.75,    0, 0.25])
    >>> y = np.array([-0.25, 0.75, -0.5,    0])
    >>> add_noise(x,y,SNR = 0, normalization_mode = 'scale')
    array([ 0.16666667,  1.        , -0.33333333,  0.16666667])
    >>> add_noise(x,y,SNR = 0, normalization_mode = 'clip')
    array([ 0.25,  1.  , -0.5 ,  0.25])
    
    '''
    if normalization_mode not in ['scale','clip']:
        print('Invalid string entered for normalization mode! Must be either "scale" or "clip"! Function terminating.')
        return
    
    if SNR == np.NINF:
        return input_noise
    else:
        signal_power = np.mean(input_signal**2) # Calculate signal power as the expectation of its squared values.
        noise_power = np.mean(input_noise**2) # Calculate noise power as the expectation of its squared values.

        K = (signal_power/noise_power)*(10**(-SNR/10));  # Scale factor K is calculated based on desired SNR in the input.

        scaled_noise = np.sqrt(K)*input_noise # Multiply the noise by the scale factor K to obtain the desired SNR.

        noisy_signal = input_signal + scaled_noise # The output signal is just the sum of the input signal and the scaled noise...
        
        # ...subject to these constraints below.
        if np.any(np.abs(noisy_signal) > 1): # If any values of the signal lie outside of [-1,1],...
            # ...then we normalize the signal according to the desired normalisation mode
            if verbose:
                print('Combined signal + noise has elements that are outside of [-1,1]. Now normalising by ' + normalization_mode)
                
            if normalization_mode == 'scale':
                return noisy_signal/np.max(np.abs(noisy_signal)) # Multiply entire signal by its maximum absolute value to scale it to [-1,1].
            else: # Means it's 'clip' mode.
                return np.array([min(1,max(-1,i)) for i in noisy_signal]) # Set out-of-range values to -1 or 1, whichever is closer.
        else:
            if verbose:
                print('Combined signal + noise has no elements outside [-1,1]. Outputting it now...')
            return noisy_signal

#%%==================================================== audio_to_mel_magnitude_spectrogram =============================================================%%#

def audio_to_mel_magnitude_spectrogram(input_data = np.sin(np.linspace(0,440,44100)*np.pi), sr = None, n_fft = 1024, hop_length = 512, center = False,
                                       n_mels = 64, fmin = 20, fmax = 20000, ref = 1.0,
                                       plot_spectrogram = False, titles = [], figsize = (20,4), fontsize = 20, vmin = None, vmax = None,
                                       **kwargs):
    '''
    ========
     Inputs
    ========
    
    input_data : str or np.ndarray
        If a string, it is the filepath of the input data (that
        will be read by sf.read).
        If it is an np.ndarray, it should be either a (n,)- or
        (c,n)-shaped array if it is single-channel or multi-
        channel respectively, where c is the number of channels in
        the signal and n is the number of samples in each channel.
        It will represent the signal in floating point numbers
        between -1 and 1. This function will convert a (n,)-
        shaped array to a (1,n)-shaped array while running.
        Default is a one-second 440Hz sine tone sampled at 44100Hz.
    
    sr : int
        The sampling rate of the signal specified by input_data.
        If it is not specified, then the native sampling rate of
        the file in input_data will be used (if input_data is a
        string) or a default sampling rate of 44100Hz will be used
        (if input_data is an np.ndarray).
    
    n_fft : int
        The number of samples in each time window for the
        calculation of the STFT of the input signal using
        librosa.core.stft.
        
    hop_length : int
        The number of overlapping samples in each time window for
        the calculation of the STFT of the input signal using
        librosa.core.stft.
    
    center : bool
        If True, centers the window at the current time index of
        the signal before performing FFT on it. If False, does not
        center the window, i.e. the current time index IS the 
        first index of the window. This is as per the parameter in
        librosa.core.stft.
        
    n_mels : int
        The number of mel bands used for calculation of the log-
        frequency power spectrom by librosa.feature.melspectrogram
    
    fmin : float
        The minimum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    fmax : float
        The maximum frequency of the mel-filter bank used to
        calculate the log-frequency power spectrum in
        librosa.feature.melspectrogram.
    
    ref : float or callable
        The reference value or a function which returns the
        reference value for which dB computation is done in
        librosa.power_to_db.
    
    plot_spectrogram : bool
        If True, plots one spectrogram per channel in the signal
        specified in input_data. Otherwise, plots nothing.
    
    titles : list of str
        A list of c strings, where c is the number of channels in
        input_data, corresponding to the title of each mel
        spectrogram that is plotted. A default string will be
        assigned to each spectrogram if titles is an empty list.
        
    figsize : tuple of 2 int values
        If plot == True, this is the size of the spectrogram plot.
        If plot == False, this argument is ignored.
        
    fontsize : int
        The font size for the title of the plots, if plot == True.
        The font size of other elements in the plot will be 
        resized relative to this number. If plot == False, this
        argument is ignored.
    
    vmin : float or None
        The minimum value of the colour bar legend in the
        plt.clim() call. If None, the minimum of the input
        spectrogram values is used.
    
    vmax : float or None
        The maximum value of the colour bar legend in the
        plt.clim() call. If None, the maximum of the input
        spectrogram values is used.

    **kwargs : dict
        Extra keyword arguments to control the plot formatting
        with librosa.display.specshow (e.g. x_axis = 's',
        y_axis = 'mel', etc.)

    =========
     Outputs
    =========
    
    mel_spectrograms : np.ndarray
        An (n_mels, t, c)-shaped array containing the mel-
        spectrograms of each channel, where n_mels is the number
        of mel bands specified in the input argument to this
        function, t is the number of time bins in the STFT, and
        c is the number of channels in input_data.
        Hypothesis: t depends on the input argument center as
        follows:
        If center is True, then t = np.ceil(n/hop_length).
        If center is False, then t = np.floor(n/hop_length)-1

    ==============
     Dependencies
    ==============
    
    librosa, librosa.display, matplotlib.pyplot (as plt), numpy (as np), soundfile
    
    ================
     Future updates
    ================
    
    Split this function into two: One for outputting the mel spectrograms and one to plot them.
    
    '''
    ## EXCEPTION HANDLING

    if type(input_data) == str: # If the input data entered is a string,...
        input_data, native_sr = sf.read(input_data) # ...then we read the filename specified in the string.
        input_data = np.transpose(input_data) # Transpose the input data to fit the (c,n)-shape desired.
        sr = native_sr if sr == None else sr
    elif type(input_data) == np.ndarray: # else we assume it is an np.ndarray
        sr = 44100 if sr == None else sr
    else:
        print('Invalid input data type! Input data must either be a string or a numpy.ndarray. Program terminating.')
        return


    # At this point, input_data should be either a (n,)- or (c,n)-shaped array.
    if len(input_data.shape) == 1: # If it's a (n,)-shaped array,...
        input_data = np.expand_dims(input_data,0) # ...then convert it into a (1,n)-shaped array.
    
    if len(titles) == 0:
        titles = ['Log-frequency power spectrogram (channel {:d})'.format(i) for i in range(input_data.shape[0])] # The default title for each spectrogram just iterates on the channel number.
    elif input_data.shape[0] != len(titles):
        print('Number of titles does not match number of channels in input data! Program terminating.')
        return
    
    if input_data.shape[0] > input_data.shape[1]:
        print('Warning: The input data appears to have more channels than samples. Perhaps you intended to input its transpose?')

    ## CALCULATE MEL SPECTROGRAM OF ZEROTH CHANNEL

    # Firstly, calculate the short-time Fourier transform (STFT) of the signal with librosa.core.stft.
    # We typecast input_data[0] as a Fortran-contiguous array because librosa.core.stft does vectorised operations on it,
    # and numpy array slices are typically not Fortran-contiguous.
    input_stft = librosa.core.stft(y = np.asfortranarray(input_data[0]), n_fft = n_fft, hop_length = hop_length, center = center)

    # Then, calculate the mel magnitude spectrogram of the STFT'd signal with librosa.feature.melspectrogram.
    power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)

    # Convert the power spectrogram into into units of decibels with librosa.power_to_db.
    mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)

    # mel_spectrogram is an np.array. We typecast all elements to np.float32 to ensure all output data types match.
    mel_spectrogram = mel_spectrogram.astype(np.float32)


    ## PLOT MEL SPECTROGRAM OF ZEROTH CHANNEL

    if plot_spectrogram:
        plt.figure(figsize = figsize) # Sets size of plot.
        librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
        plt.title(titles[0], fontsize = fontsize)         # Set title of figure.
        plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
        plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
        plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
        plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
        plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
        plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
        plt.show()                                        # Display the actual plot on the IPython console.


    ## INITIALISE OUTPUT ARRAY (3-DIMENSIONAL) BASED ON ZEROTH CHANNEL MEL SPECTROGRAM SHAPE

    mel_spectrograms = np.zeros((mel_spectrogram.shape[0],mel_spectrogram.shape[1],input_data.shape[0]))
    mel_spectrograms[:,:,0] = mel_spectrogram # Put the zeroth channel mel spectrogram in first.


    ## CALCULATE AND PLOT MEL SPECTROGRAM OF OTHER CHANNELS, IF ANY

    for i in range(1,input_data.shape[0]): # for i in 1:(number of channels),...
        input_stft = librosa.core.stft(y = np.asfortranarray(input_data[i]), n_fft = n_fft, hop_length = hop_length, center = center)
        power_spectrogram = librosa.feature.melspectrogram(S = np.abs(input_stft)**2, sr = sr, n_mels = n_mels, hop_length = hop_length, n_fft = n_fft, fmin = fmin, fmax = fmax)
        mel_spectrogram = librosa.power_to_db(power_spectrogram, ref = ref)
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        mel_spectrograms[:,:,i] = mel_spectrogram # Put the calculated mel spectrogram as that of the ith channel.

        if plot_spectrogram:
            plt.figure(figsize = figsize) # Sets size of plot.
            librosa.display.specshow(mel_spectrogram, sr = sr, hop_length = hop_length, fmin = fmin, fmax = fmax, **kwargs)
            plt.title(titles[i], fontsize = fontsize)  # Set title of figure.
            plt.xlabel('Time/s', fontsize = fontsize)         # Add in x-axis label to graph.
            plt.xticks(fontsize = 0.7*fontsize)               # Set font size for x-axis ticks (i.e. the numbers at each grid line).
            plt.ylabel('Frequency/Hz', fontsize = fontsize)   # Add in y-axis label to graph.
            plt.yticks(fontsize = 0.7*fontsize)               # Set font size for y-axis ticks (i.e. the numbers at each grid line).
            plt.colorbar(format='%+3.1f dB')                  # Adds in colour bar (legend) for values in spectrogram.
            plt.clim(vmin = vmin,vmax = vmax)                 # Defines the colour bar limits.
            plt.show()                                        # Display the actual plot on the IPython console.

    return mel_spectrograms

#%%=================================================== center_padded_split_track_by_silence ============================================================%%#

def center_padded_split_track_by_silence(input_filename = None, x = [],
                                         sr = None, output_directory = '', split_length = 4, top_db = 60, tolerance_ratio = 0, verbose = False):
    """
    Function that splits track by identifying silent segments and outputs non-silent segments of split_length with at least tolerance_ratio of non-silence
    to silence. Segments are preferentially center-padded with zeroes.
    
    If a file name is specified, then the .wav file specified by that file name will be used for splitting.
    If a file name is not specified, then a nonempty numpy.ndarray of values between -1 and +1 must be specified instead.
    If no file name or nonempty numpy.ndarray is specified, then a default numpy.ndarray of 2 sets of alternating 2s white
    noise and 2s silence sampled at 44100Hz will be used as the signal.
    If both a file name and nonempty numpy.ndarray is specified, then the file name will take precedence. In other words,
    the numpy.ndarray will be ignored and the .wav file specified by the file name will be used for splitting.

    ========
     Inputs
    ========

    input_filename : str or None
        The name or path of the input file in .wav format.
        If None, then it is assumed that a numpy.ndarray is used for
        input.
        The file must contain an input signal of floating point
        numbers between -1 and +1.
    x : 1-dimensional numpy.ndarray or None
        An input signal of floating point numbers between -1 and +1.
        If None, then it is assumed that a file name is used for
        input.
        Default signal is 2 sets of alternating 1s white noise and 
        1s silence sampled at 44100Hz.
    sr : int or None
        The sampling rate of the output files that are written.
        If None, then this will be set to the sampling rate of the
        file in input_filename if it is nonempty, and 44100 if
        input_filename is nonempty.
    output_directory : str
        The output directory where the processed files will be 
        written to. If an empty string is provided, then the files
        will be outputted to the current working directory.
        Hence, the default output directory IS the current working
        directory.
    split_length : float
        The length in seconds that each output file should be.
    top_db : float
        The number of dB below a reference level of 1 that a segment
        will be considered as silence. It is exactly the same as the
        top_db parameter in librosa.effects.split.
    tolerance_ratio : float (in [0,1])
        The minimum proportion of non-silent samples in each output
        file.
    verbose : bool
        If true, prints out names of files as they are outputted
        from the buffer.
        If false, nothing is printed out.
    
    =========
     Outputs
    =========

    This function has no outputs. However, it will output files 
    of the format <input_filename>_<output_file_index>.wav to the
    directory specified in output_directory.
    
    ==============
     Dependencies
    ==============
    
    librosa, numpy (as np), os, soundfile (as sf)

    ==========
     Examples
    ==========

    >>> center_padded_split_track_by_silence(output_directory = "sample_track" + os.sep, tolerance_ratio = 0.1) # Outputs two files, both containing 2s of white noise center-padded with silence.

    ================
     Future updates
    ================
    
    None at the moment.
    
    """
    #%%== EXCEPTION HANDLING ==%%#
    if input_filename == None:
        input_filename = '' # ...we assign an empty string to input_filename to prevent errors when writing with sf.write later,...
        if sr == None:
            sr = 44100
        if len(x) == 0:
            x = np.concatenate((2*np.random.rand(88200)-1,np.zeros(88200),2*np.random.rand(88200)-1,np.zeros(88200))) # ...and set a default signal of alternating white noise and silence.
    else: 
        x, sr = librosa.load(input_filename, sr = sr, mono = True, dtype = np.float32) # Recall that librosa.load outputs both the signal and its sampling frequency.
   
    if len(output_directory) > 0 and (not os.path.exists(output_directory)): # If the output directory is not an empty string and doesn't yet exist,...
        os.makedirs(output_directory) # ...then we create it (as an empty directory of course).

    # Use librosa.effects.split to obtain non-silent intervals in x.
    intervals = librosa.effects.split(x, top_db = top_db, ref = 1, frame_length=2048, hop_length=512) # Is a 2-column array of start and end time indices.
                                                                                                      # Start indices are inclusive and end indices are exclusive,
                                                                                                       # as per regular Python syntax.
    if len(intervals) == 0: # If the entire track is silent,...
        print('Track is silent. Nothing to output. Program terminating.')
        return # ...then there's nothing to write.

    if int(split_length*sr) < 1: # If the desired buffer size is less than the time between each sample,...
        print('Split lengths are too small for sampling frequency. Increase either quantity and try again. Program terminating.')
        return # ...then it's impossible to output anything and there's nothing to write.
    
    if tolerance_ratio != np.min([np.max([tolerance_ratio,0]),1]): # If the tolerance ratio is outside of [0,1],...
        tolerance_ratio = np.min([np.max([tolerance_ratio,0]),1]) # ...then set it to the bound that is closer to the provided value.
        print('Tolerance ratio out of bounds. Setting it to {:d}.'.format(tolerance_ratio))
        
    #%%== MAIN CODE STARTS HERE ==%%#
    # If the code makes it this far, then intervals is nonempty and there is at least one nonsilent segment in the signal.
    # The 'for' loop marks each element of the signal as silent (False) or non-silent (True).
    sample_is_nonsilent = np.full(len(x),False) # Initialisation.
    for i in range(intervals.shape[0]):
        sample_is_nonsilent[intervals[i,0]:intervals[i,1]] = np.full(intervals[i,1]-intervals[i,0],True)

    buffer_left_position = 0 # Initialise buffer left and right positions at the start of the signal
    buffer_right_position = 0 # The left position will eventually be included and the right position will not be included, as per Python convention.
    buffer_size = int(split_length*sr) # Initialise the buffer size as the number of samples of each output file.
    buffer_nonsilent_samples = 0 # Initialise variable to track numbers of non-silent samples in buffer.
    output_file_index = 1 # This is the current index of the file that will be output next from the buffer. Initialised to 1.
    output_filename = (output_directory # Start with the specified output directory, then follow with the filename
                      + input_filename.split(os.sep)[-1][:-4] # Remove the last four characters of the input filename because it is '.wav'. We use split on os.sep in case a path was provided.
                      + '_{:04d}.wav' # Add a placeholder for the counter to the filename
                      if len(input_filename) > 1 else 
                      output_directory # Start with the specified output directory, then follow with the filename
                      + '{:04d}.wav') # Add a placeholder for the counter to the filename

    for curr_signal_position in range(len(x)): # Iterate backwards through all signal positions.
        # Test if the buffer is empty or full (it's possible that the buffer is neither empty nor full --- when it's partially filled)
        buffer_is_empty = True if buffer_right_position - buffer_left_position == 0 else False
        buffer_is_full = True if buffer_right_position - buffer_left_position == buffer_size else False

        if buffer_is_empty: # (1) If the buffer is empty,...
            if sample_is_nonsilent[curr_signal_position]: # (1A) ...and if sample is nonsilent at the current signal position,...
                buffer_right_position += 1 # ...then increment the buffer's right position (i.e. add the current sample to the buffer),...
                buffer_nonsilent_samples += 1 # ...and increment the counter for nonsilent samples in the buffer.
            else: # (1B) ...and if the the sample if silent at the current signal position,...
                buffer_left_position += 1 # ...then move both buffer positions to the right by one unit (i.e. ignore the current sample).
                buffer_right_position += 1
        elif buffer_is_full: # (2) If the buffer is full,...
            # ...then we generate the samples in the buffer with librosa.utils.pad_center (which uses np.pad and pads with 0s by default),...
            #    (Note that buffer_left_position may be < 0 if there are too many silent samples at the start of the track, so we take the 
            #     max of it and 0 to get the values out from the signal x before padding. Similar reasoning for the min of len(x) and 
            #     buffer_right_position. Also, if the code makes it here, then there are at most buffer_size samples and librosa.utils.
            #     pad_center should not throw a ParameterError)
            buffer = librosa.util.pad_center(x[max(0,buffer_left_position):min(len(x),buffer_right_position)], buffer_size)
            
            # ...output the buffer (if it is at least tolerance_ratio nonsilent),...
            if buffer_nonsilent_samples/buffer_size >= tolerance_ratio:
                sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                         data = buffer, samplerate = sr)
                if verbose:
                    print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/buffer_size, output_filename.format(output_file_index)))
                output_file_index += 1 # Increment the output file index for the next file to be outputted from the buffer.
            elif verbose:
                print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/buffer_size))
            
            buffer_nonsilent_samples = 0 # ...reset the number of nonsilent samples,...
            
            # ...and do the processing for the current sample.
            if sample_is_nonsilent[curr_signal_position]: # (2A) If sample is nonsilent at the current signal position,...
                # ...then reset the buffer positons to the current signal position and add the current sample to the buffer),...
                buffer_left_position = curr_signal_position
                buffer_right_position = curr_signal_position + 1 
                buffer_nonsilent_samples += 1 # ...and increment the counter for nonsilent samples in the buffer.
            else: # (2B) ...and if the the sample if silent at the current signal position,...
                # ...then reset the buffer positions to the next signal position (i.e. ignore the current sample).
                buffer_left_position = curr_signal_position + 1
                buffer_right_position = curr_signal_position + 1
        else: # (3) If the buffer is partially filled,...
            if sample_is_nonsilent[curr_signal_position]: # (3A) If sample is nonsilent at the current signal position,...
                buffer_right_position += 1 # ...then add the current sample to the buffer,
                buffer_nonsilent_samples += 1 # ...and increment the counter for nonsilent samples in the buffer.
            else: # (3B) ...and if the the sample if silent at the current signal position
                buffer_right_position += 1 # ...then add the current sample to ther buffer,
                buffer_is_full = True if buffer_right_position - buffer_left_position == buffer_size else False # ...check if the buffer is now full,...
                buffer_left_position = buffer_left_position if buffer_is_full else buffer_left_position - 1# ...and move the left position to the left (to center-pad the signal) if the buffer is not full yet.

    # Generate and output any remaining samples in the buffer (if it is at least tolerance_ratio nonsilent).
    buffer = librosa.util.pad_center(x[max(0,buffer_left_position):min(len(x),buffer_right_position)], buffer_size)  
    
    if buffer_nonsilent_samples/buffer_size >= tolerance_ratio:
        sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                 data = buffer, samplerate = sr)
        if verbose:
            print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/buffer_size, output_filename.format(output_file_index)))
    elif verbose:
        print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/buffer_size))

#%%============================================================ fisher_yates_shuffle ===================================================================%%#

def fisher_yates_shuffle(*mutables):
    '''
    Performs an in-place unison Fisher-Yates shuffle of
    any tuple of mutables (e.g. numpy arrays, lists,
    etc.) of identical length in the first dimension,
    and returns the sequence of swaps that resulted in
    the in-place shuffle performed.
        
    ========
     Inputs
    ========
    
    *mutables : tuple
        A tuple of arrays that the unison shuffle is to
        be performed on.
    
    =========
     Outputs
    =========

    jays : list of n-1 int
        The sequence of random integers j that is
        generated in order to perform the Fisher-Yates
        shuffle. See the 'Idea' section for more
        elaboration.
        
    ======
     Idea
    ======
    
    The Fisher-Yates shuffle is a memory-efficient
    shuffling algorithm, which can be described in
    pseudocode as follows, given that a zero-indexed
    array a containing n elements is to be shuffled:
    
    for i from n-1 down to 1 do:
        j <- random integer in range [0,i]
        swap a[j] and a[i]
        
    ==============
     Dependencies
    ==============
    
    numpy (as np)
    
    ==========
     Examples
    ==========
    
    # Shuffle some training data and training labels
    # in unison, such that the shuffled data and 
    # labels still match each other.
    >>> np.random.seed(2020)
    >>> train_data   = ['a','b','c','d','e']
    >>> train_labels = [ 4,  2,  6,  9, -1 ]
    >>> order = fisher_yates_shuffle(train_data,
                                     train_labels)
    >>> order
    [0, 0, 2, 1]
    >>> train_data
    ['d', 'b', 'c', 'e', 'a']
    >>> train_labels
    [9, 2, 6, -1, 4]

    # To replicate the shuffling with the output,
    # we can do the following:
    >>> train_data   = ['a','b','c','d','e']
    >>> order = [0, 0, 2, 1]
    >>> for i in range(len(order)):
            train_data[len(order)-i], train_data[order[i]] = train_data[order[i]], train_data[len(order)-i]
    >>> train_data
    ['d', 'b', 'c', 'e', 'a']
    
    # To just generate an arbitrary order of indices 
    # for a Fisher-Yates shuffle, run the function with
    # any mutable object of the desired length. This can
    # be useful in cases where the mutables to be
    # shuffled are too large for memory and need to be
    # processed separately outside this function.
    >>> np.random.seed(2020)
    >>> rand_perm = fisher_yates_shuffle(np.zeros(10))
    [0, 8, 3, 6, 3, 3, 1, 0, 1]
    
    # BEWARE: Slices of lists are not modifed by
    # functions! If you want to perform a shuffle
    # on a list slice, either use the output of this
    # function or create a new list that can be
    # passed into this function in its entirety.
    >>> data = [0, 1, 2, 3, 4, 5]
    >>> fisher_yates_shuffle(data[1:])
    >>> data
    [0, 1, 2, 3, 4, 5]
    >>> data_new = data[1:]
    >>> np.random.seed(2020)
    >>> fisher_yates_shuffle(data_new)
    >>> data_new
    [4, 2, 3, 5, 1]
    '''
    # EXCEPTION HANDLING
    
    first_dimension_length = len(mutables[0])
    for k in range(1,len(mutables)):
        if len(mutables[k]) != first_dimension_length:
            print('Not all mutables entered have same first dimension length. Program terminating.')
            return []
        
    # PERFORM UNISON SHUFFLE
    
    jays = []
    for i in range(first_dimension_length-1, 0, -1): 

        # Pick a random index from 0 to i inclusive.
        j = np.random.randint(i+1)
        
        # Add this index to jays for tracking.
        jays.append(j)

        # Swap the ith element of each mutable in mutables with the element at random index j in unison.
        for k in range(len(mutables)):
            mutables[k][i], mutables[k][j] = mutables[k][j], mutables[k][i]

    return jays

#%%============================================================ generate_framewise_gt ==================================================================%%#

def generate_framewise_gt(csv_filepath_read = '', ground_truth_intervals = [], header = False, csv_filepath_write = '',
                          n_samples = 2*44100, window_size = 512, 
                          classes = ['ambience','cough','crying','falling','shout','speech'],
                          multilabel = False, plot_gt = False, return_gt = True, **kwargs):
    """
    Generates framewise ground-truth .csv files given
    samplewise ground-truth .csv files. Works for
    single-label and multi-label, single-class and
    multi-class ground truth tasks.   
    
    ========
     Inputs
    ========
    csv_filepath_read : str
        The path to the .csv file, that when loaded by
        read_csv, returns a list of lists of 3 strings,
        in the same format as ground_truth_intervals.
        
    ground_truth_intervals : list of lists of 3 strings
        A list containing n lists, each containing 3
        strings, where n is the number of sound events
        detected, the zeroth string denotes the start
        index of the event, the first string denotes
        the end index of the event, and the 2nd string
        denotes the class of the event. Start and end
        indices work Python-style. Argument will be
        overridden by csv_filepath if csv_filepath is
        nonempty.
    
    header : bool
        If True, removes the first element of the
        loaded .csv file in csv_filepath or
        ground_truth_intervals before processing and 
        treats it as a header. If False, does nothing.
    
    csv_filepath_write : str
        The path to write the framewise ground-truth
        labels generated by this function.
    
    n_samples : int
        The number of samples in the audio track that
        has csv_filepath/ground_truth_intervals as its
        ground-truth labels.
    
    window_size : int
        The number of samples in each prediction window.
        There is, and should not be, any overlap
        between windows.   
        
    classes : list of str
        A list containing k strings, where k is the 
        number of classes for the prediction model.
        Each string should contain the name of an
        individual class.
        
    multilabel : bool
        If True, then csv_filepath or
        ground_truth_intervals will be assumed to be
        general binary matrices instead of one-hot
        encoded matrices. This corresponds to a multi-
        label classification ground truth. If False,
        then they will be assumed to be one-hot
        encoded matrices. This corresponds to a single-
        label classification ground truth.
    
    plot_gt : bool
        If True, plots the framewise ground truth label
        matrix with seaborn.heatmap(). If False, plots
        nothing.
        
    return_gt : bool
        If True, will return ground_truth_labels_by_
        frame as an output argument. If False, returns
        None.
        
    **kwargs : dict
        Other keywords arguments to pass to 
        load_predictions for plotting.
        
    =========
     Outputs
    =========
    
    ground_truth_labels_by_frame : np.ndarray
        A (len(classes), n_frames) np.ndarray containing
        the binary 0/1 labels for each frame of the
        ground truth.
        
    ==========
     Examples
    ==========

    E.g.1 The intervals giving a samplewise (binary)
    ground-truth matrix of
                    111000011000
                    000101111000
    would correspond to an output of
                    1 1 0 1 1 0 
                    0 1 1 1 1 0 
    if the window size is 2, as shown in the following
    function call.
    
    >>> generate_framewise_gt(ground_truth_intervals = [[0,3,'a'],[7,9,'a'],[3,4,'b'],[5,9,'b']], n_samples = 12, window_size = 2, classes = ['a','b'], multilabel = True)
    array([[1., 1., 0., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.]])

    E.g.2 The intervals giving a samplewise (one-hot)
    ground-truth matrix of
                    111100
                    000010
                    000001
    would correspond to an output of
                    1  1 
                    0  0
                    0  0
    if the window size is 3, as shown in the following
    function call.
    
    >>> generate_framewise_gt(ground_truth_intervals = [[0,4,'a'],[4,5,'b'],[5,6,'c']], n_samples = 6, window_size = 3, classes = ['a','b','c'])
    array([[1., 1.],
           [0., 0.],
           [0., 0.]])
    
    ==============
     Dependencies
    ==============
    
    csv, matplotlib.pyplot (as plt), numpy (as np),
    seaborn
    """
    ## EXCEPTION HANDLING
    if len(csv_filepath_read) != 0:
        ground_truth_intervals = read_csv(csv_filepath_read)
    elif len(ground_truth_intervals) == 0:
        print('Error: No ground truth file or array specified.')
        return
    
    if header:
        ground_truth_header = ground_truth_intervals.pop(0)
    
    ## GENERATE SAMPLEWISE GROUND TRUTH LABEL MATRIX
    ground_truth_labels = np.zeros(shape = (len(classes), n_samples)) # Will be a binary matrix with 1 if sound event present in current sample, 0 otherwise.
    for row in ground_truth_intervals:
        ground_truth_labels[classes.index(row[2]), int(row[0]):int(row[1])] = np.ones(shape = (int(row[1])-int(row[0]),)) 
    
    ## INITIALISE PARAMETERS TO GENERATE FRAMEWISE GROUND TRUTH LABEL MATRIX (i.e. ground_truth_labels_by_frame)
    n_frames = int(np.ceil(n_samples/window_size))                            # Number of frames in track
    ground_truth_labels_by_frame = np.zeros(shape = (len(classes), n_frames)) # Will be a binary matrix with 1 if sound event present in current frame, 0 otherwise.
    Lpointer = 0                                                              # Tracks left side of window in units of sample indices.
    Rpointer = window_size                                                    # Tracks right side of window in units of sample indices.
    
    ## GENERATE FRAMEWISE GROUND TRUTH LABEL MATRIX
    for current_frame in range(n_frames):                                                                                                                                        # Add hop_length to ensure that the length of the ground truth labels matches with the length of the prediction matrices.
        current_ground_truth_label = np.zeros(shape = (len(classes),), dtype = int)                                                                                              # Force datatype to int to prevent floating point errors later on.
        if multilabel:                                                                                                                                                           # In a multi-label classification task, labels the current frame with the class iff it has more samples than half the window size.
            ground_truth_labels_by_frame[:, current_frame] = (np.sum(ground_truth_labels[:,Lpointer:int(min(Rpointer,n_samples))], axis = 1) >= int(np.ceil(window_size/2))) + 0 # +0 forces typecast to int without changing values.
        else:                                                                                                                                                                    # In a single-label classification task, label the current frame with the class containing the most samples.
            ground_truth_class_index = np.argmax( np.sum(ground_truth_labels[:,Lpointer:int(min(Rpointer,n_samples))], axis = 1) )                                               # In event of tie, np.argmax only returns the first value matching the maximum.
            ground_truth_labels_by_frame[ground_truth_class_index, current_frame] = 1
        Lpointer += window_size                                                                                                                                                  # Shift Lpointer and Rpointer by the window size.
        Rpointer += window_size                                                                                                                                                  # Rpointer may exceed track length in samples! So we take min(Rpointer,n_samples) when calling indices in ground_truth_labels

    ## PLOT AND SAVE FRAMEWISE GROUND TRUTH LABEL MATRIX
    if plot_gt:
        load_predictions(predictions = ground_truth_labels_by_frame, binary = False, plot_predictions = plot_gt, yticklabels = classes, **kwargs)
    
    if len(csv_filepath_write) > 0:
        write_csv(ground_truth_labels_by_frame.transpose(), csv_filepath_write, newline = '\n')
        
    if return_gt:
        return ground_truth_labels_by_frame

#%%===================================================== left_padded_split_track_by_silence ============================================================%%#

def left_padded_split_track_by_silence(input_filename = None, x = [], sr = None, output_directory = '',
                                       split_length = 4, top_db = 60, tolerance_ratio = 0, verbose = False):
    """
    Function that splits track by identifying silent segments and outputs non-silent segments of split_length with at least tolerance_ratio of non-silence
    to silence. Segments are preferentially left-padded with zeroes.
    
    The method used is:

    1. Initialise an empty buffer B of split_length seconds to store any output data. 
    2. Given an n-by-2 matrix intervals from librosa.effects.split, split the data into intervals labelled "silence" and "non-silence" alternatingly.
       There should be n non-silent intervals and up to n+1 silent intervals (depending on whether the start and end of the track is considered non-silent.
       For the purposes of this algorithm, we assume n non-silent intervals and n+1 silent intervals for a total of 2n+1 intervals.
    3. For each interval I_k, where k iterates through {-1, -2,... ,-2n-1} (i.e. backward in time) starting from the latest non-silent interval,
        1. If I_k is silent, keep adding samples from I_k until either:
            1. The buffer B is full: Then discard all remaining samples from I_k, output the buffer, and move to I_{k-1}.
            2. There are no more samples to add to B: Then move to I_{k-1}.
        2. If I_k is non-silent, keep adding samples from I_k until either:
            1. The buffer B is full: Then output the buffer, and continue adding samples from I_k into B.
               Repeat this action until there are no more samples to add to B.
            2. There are no more samples to add to B: Then move to I_{k-1}.
    4. At the end, left-pad the buffer with silence and output it.
    5. Output the buffer if there is at least tolerance_ratio of non-silence to silence, else discard it.

    If a file name is specified, then the .wav file specified by that file name will be used for splitting.
    If a file name is not specified, then a nonempty numpy.ndarray of values between -1 and +1 must be specified instead.
    If no file name or nonempty numpy.ndarray is specified, then a default numpy.ndarray of 2 sets of alternating 2s white
    noise and 2s silence sampled at 44100Hz will be used as the signal.
    If both a file name and nonempty numpy.ndarray is specified, then the file name will take precedence. In other words,
    the numpy.ndarray will be ignored and the .wav file specified by the file name will be used for splitting.

    ========
     Inputs
    ========

    input_filename : str or None
        The name or path of the input file in .wav format.
        If None, then it is assumed that a numpy.ndarray is used for
        input.
        The file must contain an input signal of floating point
        numbers between -1 and +1.
    x : 1-dimensional numpy.ndarray or None
        An input signal of floating point numbers between -1 and +1.
        If None, then it is assumed that a file name is used for
        input.
        Default signal is 2 sets of alternating 1s white noise and 
        1s silence sampled at 44100Hz.
    sr : int or None
        The sampling rate of the output files that are written.
        If None, then this will be set to the sampling rate of the
        file in input_filename if it is nonempty, and 44100 if
        input_filename is nonempty.
    output_directory : str
        The output directory where the processed files will be 
        written to. If an empty string is provided, then the files
        will be outputted to the current working directory.
        Hence, the default output directory IS the current working
        directory.
    split_length : float
        The length in seconds that each output file should be.
    top_db : float
        The number of dB below a reference level of 1 that a segment
        will be considered as silence. It is exactly the same as the
        top_db parameter in librosa.effects.split.
    tolerance_ratio : float (in [0,1])
        The minimum proportion of non-silent samples in each output
        file.
    verbose : bool
        If true, prints out names of files as they are outputted
        from the buffer.
        If false, nothing is printed out.
    
    =========
     Outputs
    =========

    This function has no outputs. However, it will output files 
    of the format <input_filename>_<output_file_index>.wav to the
    directory specified in output_directory.
    
    ==============
     Dependencies
    ==============
    
    librosa, numpy (as np), os, soundfile (as sf)

    ==========
     Examples
    ==========

    >>> left_padded_split_track_by_silence(output_directory = "sample_track" + os.sep) # Outputs two files, both containing 2s of silence followed by 2s of white noise.

    ================
     Future updates
    ================
    
    None at the moment
    
    """
    #%%== EXCEPTION HANDLING ==%%#
    if input_filename == None:
        input_filename = '' # ...we assign an empty string to input_filename to prevent errors when writing with sf.write later,...
        if sr == None:
            sr = 44100
        if len(x) == 0:
            x = np.concatenate((2*np.random.rand(88200)-1,np.zeros(88200),2*np.random.rand(88200)-1,np.zeros(88200))) # ...and set a default signal of alternating white noise and silence.
    else: 
        x, sr = librosa.load(input_filename, sr = sr, mono = True, dtype = np.float32) # Recall that librosa.load outputs both the signal and its sampling frequency.
   
    if len(output_directory) > 0 and (not os.path.exists(output_directory)): # If the output directory is not an empty string and doesn't yet exist,...
        os.makedirs(output_directory) # ...then we create it (as an empty directory of course).

    # Use librosa.effects.split to obtain non-silent intervals in x.
    intervals = librosa.effects.split(x, top_db = top_db, ref = 1, frame_length=2048, hop_length=512) # Is a 2-column array of start and end time indices.
                                                                                                      # Start indices are inclusive and end indices are exclusive,
                                                                                                       # as per regular Python syntax.
    if len(intervals) == 0: # If the entire track is silent,...
        print('Track is silent. Nothing to output. Program terminating.')
        return # ...then there's nothing to write.

    if int(split_length*sr) < 1: # If the desired buffer size is less than the time between each sample,...
        print('Split lengths are too small for sampling frequency. Increase either quantity and try again. Program terminating.')
        return # ...then it's impossible to output anything and there's nothing to write.
    
    if tolerance_ratio != np.min([np.max([tolerance_ratio,0]),1]): # If the tolerance ratio is outside of [0,1],...
        tolerance_ratio = np.min([np.max([tolerance_ratio,0]),1]) # ...then set it to the bound that is closer to the provided value.
        print('Tolerance ratio out of bounds. Setting it to {:d}.'.format(tolerance_ratio))
        
    #%%== MAIN CODE STARTS HERE ==%%#
    # If the code makes it this far, then intervals is nonempty and there is at least one nonsilent segment in the signal.
    # The 'for' loop marks each element of the signal as silent (False) or non-silent (True).
    sample_is_nonsilent = np.full(len(x),False) # Initialisation.
    for i in range(intervals.shape[0]):
        sample_is_nonsilent[intervals[i,0]:intervals[i,1]] = np.full(intervals[i,1]-intervals[i,0],True)
        
    buffer = np.zeros(int(split_length*sr)) # Initialise empty buffer of split_length seconds.
                                            # If the code makes it this far, then buffer is nonempty and is at least one sample long.
                                            # We typecast to int in case split_length*sr is not an integer.
    buffer_position = -1 # Initialise buffer position at the rightmost (last) index.
    buffer_nonsilent_samples = 0 # Initialise variable to track numbers of non-silent samples in buffer.
    output_file_index = 1 # This is the current index of the file that will be output next from the buffer. Initialised to 1.
    output_filename = (output_directory # Start with the specified output directory, then follow with the filename
                      + input_filename.split(os.sep)[-1][:-4] # Remove the last four characters of the input filename because it is '.wav'. We use split on os.sep in case a path was provided.
                      + '_{:04d}.wav' # Add a placeholder for the counter to the filename
                      if len(input_filename) > 1 else 
                      output_directory # Start with the specified output directory, then follow with the filename
                      + '{:04d}.wav') # Add a placeholder for the counter to the filename

    for curr_signal_position in reversed(range(-len(x),0)): # Iterate backwards through all signal positions.
        # Test if the buffer is empty or full (it's possible that the buffer is neither empty nor full --- when it's partially filled)
        buffer_is_empty = True if buffer_position == -1 else False
        buffer_is_full = True if buffer_position < -len(buffer) else False

        if buffer_is_empty: # (1) If the buffer is empty,...
            if sample_is_nonsilent[curr_signal_position]: # (1A) ...and if sample is nonsilent at the current signal position,...
                buffer[buffer_position] = x[curr_signal_position] # ...then add the current sample to the buffer,...
                buffer_nonsilent_samples += 1 # ...increment the counter for nonsilent samples in the buffer,...
                buffer_position -= 1 # ...and move the buffer position by one unit to the left.
            else: # (1B) ...and if the the sample if silent at the current signal position,...
                continue # ...then do nothing (i.e. we discard the current sample).
        elif buffer_is_full: # (2) If the buffer is full,...
            if sample_is_nonsilent[curr_signal_position]: # (2A) ...and if sample is nonsilent at the current signal position,...
                # ...then we output the buffer (if it is at least tolerance_ratio nonsilent),...
                if buffer_nonsilent_samples/len(buffer) >= tolerance_ratio:
                    sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                             data = buffer, samplerate = sr)
                    if verbose:
                        print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/len(buffer), output_filename.format(output_file_index)))
                    output_file_index += 1 # Increment the output file index for the next file to be outputted from the buffer.
                elif verbose:
                    print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/len(buffer)))
                buffer = np.zeros(int(split_length*sr)) # ...reset the buffer to zero to restart adding samples to the buffer,...
                buffer_position = -1 # ...reset the buffer position to the beginning,...
                buffer_nonsilent_samples = 0 # ...reset the counter for nonsilent samples in the buffer,...
                buffer[buffer_position] = x[curr_signal_position] # ...add the current sample to the buffer,...
                buffer_position -= 1 # ...and move the buffer position by one unit to the left.
                # Note that while these steps may seem convoluted, we need the code to work with any buffer size, including a buffer of size 1.
            else: # (2B) ...and if the the sample if silent at the current signal position,...
                continue # ...then do nothing (i.e. we discard the current sample).
        else: # (3) If the buffer is partially filled,...
            buffer[buffer_position] = x[curr_signal_position] # ...then add the current sample to the buffer (regardless of whether it's silent),...
            if sample_is_nonsilent[curr_signal_position]:
                buffer_nonsilent_samples += 1 # ...increment the counter for nonsilent samples in the buffer if the current sample is nonsilent,...
            buffer_position -= 1 # ...and move the buffer position by one unit to the left.

    # Output any remaining samples in the buffer (if it is at least tolerance_ratio nonsilent).
    if buffer_nonsilent_samples/len(buffer) >= tolerance_ratio:
        sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                 data = buffer, samplerate = sr)
        if verbose:
            print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/len(buffer), output_filename.format(output_file_index)))
    elif verbose:
        print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/len(buffer)))

#%%================================================================= load_predictions ==================================================================%%#

def load_predictions(predictions = np.random.rand(6,5), csv_path = '', permutation = [], binary = False, plot_predictions = False,
                     yticklabels = ['ambience','cough','crying','falling','shout','speech'], figsize = (30,10), title = 'Prediction plot', **kwargs):
    """
    Loads and plots predictions from a .csv file.
    
    ========
     Inputs
    ========
    
    predictions : np.ndarray
        A (k, n) array of predictions over k classes
        and n samples. Will be ignored if csv_path is
        a non-empty string
    
    csv_path : str
        A string containing the filepath of the .csv
        file with n rows of k entries each, that when
        loaded, will be a (k, n) array of predictions over
        k classes and n samples. Will override any array
        entered in predictions if this is a nonempty
        string.
    
    permutation : list of int
        A list of length k, where k is the number of
        classes, that specifies the order to permute
        the rows of the np.ndarray predictions.
        If left empty, or if the list is not of length
        k, then no permutation is performed.
        
    binary : bool
        If True, converts predictions into 0-1 labels
        based on argmax before plotting and returning.
        If False, plots and returns predictions as raw
        class probabilities.
    
    plot_predictions : bool
        If True, uses seaborn.heatmap to plot the
        predictions (assumed to be a time series).
        If False, plots nothing.
    
    yticklabels : list of str
        The y-axis labels for the seaborn heatmap.
        We should have len(yticklabels) == k, where k
        is the number of classes.
        
    figsize : tuple of int
        The x- and y-dimensions of the figure that
        the heatmap will be plotted into.
    
    title : str
        The title of the seaborn heatmap.
    
    **kwargs : dict
        Additional keyword arguments to be passed to
        seaboarn.heatmap().

    =========
     Outputs
    =========
    
    predictions : np.ndarray
        A (k, n) array of predictions over k classes
        and n samples.    
    
    ==============
     Dependencies
    ==============
    
    csv, matplotlib.pyplot (as plt), numpy (as np),
    seaborn
    
    useful_functions (our own user-written module)
    
    ==========
     Examples
    ==========
    
    >>> x = load_predictions(plot_predictions = True)
    # Will plot a heatmap of a randomly-generated
    # matrix.
    
    =======
     Notes
    =======
    
    For Sathish and Rishabh's old model, we should set
    permutation = [5, 1, 2, 0, 3, 4]

    """
    if len(csv_path) > 0:
        predictions = read_csv(csv_path, newline='\n') # This loads predictions as a list.
        predictions = np.array(predictions, dtype = np.float64).transpose() # We typecast it as a numpy array for easier manipulation.
                                                                            # Must typecast to np.float32 or np.float64 to prevent reading exponent bits wrongly (e.g. 5e-5 will be read as '5' and not '0.00005' if we don't force dtype = np.float32 or np.float64)
    
    if binary: # Convert class probabilities into binary vectors by argmax (this is a single-class classification problem)
        for sample_index in range(predictions.shape[1]):
            class_label = np.argmax(predictions[:,sample_index])
            for class_index in range(predictions.shape[0]):
                predictions[class_index,sample_index] = 1 if class_index == class_label else 0
                
    if len(permutation) == predictions.shape[0]:
        predictions = predictions[permutation,:]

    if plot_predictions:
        plt.figure(figsize=figsize)
        plt.title(title)
        ax = seaborn.heatmap(predictions, yticklabels = yticklabels, **kwargs)
        ax.set_yticklabels(yticklabels, verticalalignment = 'center')

    return predictions

#%%================================================================ offline_prediction =================================================================%%#

def offline_prediction(model = '', track = '', mic_noise = '', csv_path = '',
                       style = 'kenneth', buffer_size = 2*44100, hop_length = 512,
                       linus_mean = 0, linus_sd = 1, moving_average = 0, average_mode = 'mean',
                       output_features = False, feature_path = 'features.npy', verbose = False, **kwargs):
    """
    Makes offline predictions as though track was
    fed into model in an online inference engine.
    
    E.g. with each - representing a hop length of 512
    samples, 0 representing zero-padding in the
    signal, and ^--88200--^ representing a moving
    window of 88200 samples that the spectrogram is
    taken from.
    
    0000000000--------------- (prediction for frame 0)
    ^--88200--^
    
    0000000000--------------- (prediction for frame 1)
     ^--88200--^
     
    etc...

    0000000000--------------- (prediction for frame 14)
                  ^--88200--^
    
    ========
     Inputs
    ========
    
    model : keras.engine.sequential.Sequential,
            keras.engine.training.Model, or str
        The model that will be used to make predictions
        based on the recorded track in track. If a str
        is entered, it is assumed to be the path where
        the .h5 model can be loaded. Otherwise, it is
        assumed to be a pre-loaded model already.
        As a standard, SNR and Kenneth's models have the
        keras.engine.sequential.Sequential type, whereas
        Linus' models have keras.engine.training.Model
        as the type.
    
    track : np.ndarray or str
        The recorded track that predictions will be made
        on. If a str is entered, it is assumed to be the
        path where the .wav file containing the track
        can be loaded. Otherwise, it is assumed to be an
        np.ndarray with shape (n,), where n is the
        number of samples in the track.
        
    mic_noise : np.ndarray or str
        The values that will be placed in the buffer
        when there is no signal going through the
        inference engine (usually only happens at the
        start and end of inference). If a str is
        entered, it is assumed to be the path where the
        .wav file containing the track can be loaded.
        If an empty string is entered, it will be
        initialised as np.zeros(shape = (buffer_size))
        Otherwise, it is assumed to be an np.ndarray
        with shape (m,), where m >= buffer_size. If m >
        buffer_size, then the first buffer_size samples
        will be used at the start of the inference
        and the last buffer_size samples will be used
        at the end of the inference.
    
    csv_path : str
        The path of the .csv file that the predictions
        will be saved to. If empty, then a default path
        will be provided.
    
    style : str in ['kenneth', 'snr', 'linus']
        If 'kenneth', then no extra processing will be 
        done to track when it is loaded. If 'snr', then
        track will be multiplied by 10, as per the
        processing done by Sathish and Rishabh in their
        inference engine. If 'linus', then each spectro-
        gram in the buffer (spectrogram_of_buffer) will
        be standardised by the given mean and standard
        deviation vectors in linus_mean and linus_sd.

    buffer_size : int
        The size of the (audio) buffer, in number of
        samples, that predictions will be made upon with
        model.
    
    hop_length : int
        The hop length, in number of samples, that the
        buffer will move for each prediction frame.
        
    linus_mean : np.ndarray
        An (n_mels, 1) np.ndarray that contains the mean
        values of each mel frequency band to standardise
        each spectrogram in the buffer by. Note that
        n_mels is one of the **kwargs for audio_to_mel_
        spectrogram. Only used if style == 'linus'.
        
    linus_sd : np.ndarray
        An (n_mels, 1) np.ndarray that contains the
        standard deviation values of each mel frequency
        band to standardise each spectrogram in the
        buffer by. Note that n_mels is one of the
        **kwargs for audio_to_mel_spectrogram. Only used
        if style == 'linus'.
        
    moving_average : int
        The number of past prediction frames that will
        be averaged over together with the prediction of
        the current frame in order to make the final
        prediction of the current frame.
        If 0, then no average is taken (i.e. the
        prediction for the current frame doesn't depend
        on the predictions of previous frames).
    
    average_mode : str in ['mean','median']
        The mode in which the average is taken if
        moving_average != 0.
    
    output_features : bool
        If True, will write the sequence of
        spectrograms in the buffer that were used to
        generate the predictions into the path specified
        in feature_path as a .npy file. If False,
        nothing is written.
    
    feature_path : str
        The path to save the outputted sequence of
        spectrograms if output_features == True. Will be
        ignored if output_features == False.
    
    verbose : bool
        If True, prints status of predictions as they
        are made. If False, prints nothing.
    
    **kwargs : dict
        Extra keyword arguments to control parameters in
        autio_to_mel_magnitude_spectrogram.
    
    =========
     Outputs
    =========
    
    predictions.transpose() : np.ndarray
        A (k,n_frames) array with the (i,j)-th element
        containing the predicted probability of the jth
        frame belonging to the ith class, where k is 
        the number of classes and n_frames is the
        number of frames in the raw audio track (NOT
        SAMPLES!)
        
    A .csv file will also be outputted to csv_path,
    with n_frames rows and k entries per row,
    corresponding exactly to the values in predictions
    
    ==============   
     Dependencies 
    ==============
    csv, keras (2.2.4), librosa, numpy (as np),
    soundfile (as sf), tensorflow (1.14.0)
    
    useful_functions (our own user-written module)
    
    ================
     Future updates
    ================
    
    Change the function so that it outputs one csv file per channel of data.
    
    """

    ## EXCEPTION HANDLING
    if moving_average < 0 or int(moving_average) != moving_average:
        print('Error: moving_average must be a non-negative integer!')
        return
    
    if average_mode not in ['mean','median']:
        print('Error: average_mode must be \'mean\' or \'median\'!')
        return
    
    if buffer_size < 1 or int(buffer_size) != buffer_size:
        print('Error: buffer_size must be a positive integer!')
        return
    
    if hop_length < 1 or int(hop_length) != hop_length:
        print('Error: hop_length must be a positive integer!')
        return
        
    if type(model) == str:
        if len(csv_path) == 0:
            csv_path = '.'.join(model.split(os.sep)[-1].split('.')[:-1]) + '_{}_frame_moving_average.csv'.format(moving_average) # csv_path will follow model name
        model = keras.models.load_model(model, compile = False) # Declare compile = False because we're not training the model.
    elif type(model) not in [keras.engine.sequential.Sequential, keras.engine.training.Model]:
        print('Error: model is not a keras.engine.sequential.Sequential, keras.engine.training.Model, or a string.')
        return
    
    if type(track) == str:
        track, fs = sf.read(track)
    elif type(track) != np.ndarray:
        print('Error: track is not an np.ndarray or a string.')
        return
    
    if len(track.shape) != 1:
        print('Warning: track is not a single-channel file or 1-dimensional np.ndarray. Taking only 0th channel for processing...')
        track = track[:,0]
        
    if type(mic_noise) == str:
        if len(mic_noise) > 0:
            mic_noise, fs_mic_noise = sf.read(mic_noise, stop = buffer_size)
            if fs_mic_noise != fs:
                print('Error: mic_noise sampling frequency does not match track sampling frequency.')
                return
        else:
            mic_noise = np.zeros(shape = (buffer_size))
    elif type(mic_noise) != np.ndarray:
        print('Error: mic_noise is not an np.ndarray or a string.')
        return
    
    if len(mic_noise.shape) != 1:
        print('Warning: mic_noise is not a single-channel file or 1-dimensional np.ndarray. Taking only 0th channel for processing...')
        mic_noise = mic_noise[:,0]
        
    if mic_noise.shape[0] < buffer_size:
        print('Error: mic_noise has less samples than buffer_size. We need it to have at least the same number of samples to fill the buffer.')
        return
    elif mic_noise.shape[0] > buffer_size:
        print('Warning: mic_noise has more samples than buffer_size. Taking only the first buffer_size samples to fill the buffer...')
        mic_noise = mic_noise[:buffer_size]
    
    if style == 'snr':
        track = track*10 # Sathish and Rishabh's processing multiplies 10 to the microphone signal.
    elif style not in ['kenneth','linus']:
        print('Error: Style is not in ["kenneth","snr","linus"]!')
        
    if len(csv_path) == 0:
        csv_path = 'model_predictions_{}_frame_moving_average.csv'.format(moving_average)
    
    if verbose:
        print('Prediction 00% done.', end = '\r')
        
    ## GENERATE PREDICTIONS
    Lpointer = -buffer_size + hop_length
    Rpointer = hop_length
    n_frames = int(np.ceil(len(track)/hop_length)) # Make one extra prediction for the last few elements if stragglers exist.
    
    counter = 1 # Tracks percentage completion for prediction. Starts at 1%.
    for i in range(n_frames):
        if verbose and i*100/n_frames > counter:
            print('Prediction {:02d}% done.'.format(counter), end = '\r')
            counter += 1
            
        assert Lpointer < len(track) # We want to prevent errors in the if-else statement later.
            
        if Lpointer < 0 and Rpointer > len(track): # Track is too short -- asumme that track is recorded over the first buffer_size samples of mic_noise.
            buffer = np.concatenate(  (mic_noise[:(buffer_size-len(track))//2] , track , mic_noise[buffer_size-len(track)-(buffer_size-len(track))//2:])  )
        elif Lpointer < 0: # At beginning of predictions, assume missing values are the first values of mic_noise.
            buffer = np.concatenate(  (mic_noise[:buffer_size - Rpointer] , track[0:Rpointer])  )
        elif Rpointer > len(track): # At end of predictions, assume missing values are last values of mic_noise.
            buffer = np.concatenate(  (track[Lpointer:len(track)] , mic_noise[-(buffer_size - (len(track) - Lpointer)):])  )
        else:
            buffer = track[Lpointer:Rpointer]
        
        assert len(buffer) == buffer_size # We want to ensure consistency of the buffer size.
            
        spectrogram_of_buffer = audio_to_mel_magnitude_spectrogram(buffer, **kwargs)
        if style == 'linus':
            spectrogram_of_buffer = (spectrogram_of_buffer.transpose((1,0,2)) - linus_mean)/linus_sd
        prediction_of_buffer = model.predict(np.expand_dims(spectrogram_of_buffer,0))[0]
        
        if i == 0: # We can only determine the number of classes after we predict with the model at least once.
            n_classes = prediction_of_buffer.shape[0]
            predictions = np.zeros(shape = (n_frames,n_classes)) # Initialise array to store predictions for all frames.
            
            if output_features:
                n_mels = spectrogram_of_buffer.shape[0]
                n_bins = spectrogram_of_buffer.shape[1]
                features = np.zeros(shape = (n_frames,n_mels,n_bins,1)) # Initialise array to store calculated features.
            
        predictions[i,:] = prediction_of_buffer # Add current prediction to buffer
        if output_features:
            features[i,:,:,:] = spectrogram_of_buffer # Add current spectrogram to output array.
            
        Lpointer += hop_length
        Rpointer += hop_length
    
    if verbose:
        print('Prediction 100% done.')
    
    ## PERFORM MOVING AVERAGE
    if moving_average != 0:
        for i in reversed(range(len(predictions))): # Start from back to prevent overwriting required data points.
            if average_mode == 'mean':
                predictions[i] = predictions[max(0,i-moving_average):i+1].mean(axis = 0) # Take max because if we don't have enough samples to go back moving_average samples in time, then we just average up to the 0th one.
            elif average_mode == 'median':
                predictions[i] = np.median(predictions[max(0,i-moving_average):i+1], axis = 0) # Take max because if we don't have enough samples to go back moving_average samples in time, then we just average up to the 0th one.
    
    ## OUTPUT FEATURES 
    if output_features:
        np.save(feature_path, features, allow_pickle = True)
    
    ## OUTPUT CSV AND RETURN
    write_csv(predictions, csv_path, newline = '\n')
    return predictions.transpose()

#%%========================================================= plot_time_domain_signal_and_track =========================================================%%#

def plot_time_domain_signal_and_track(x = np.sin(np.linspace(0,440,44100)*np.pi) , sr = 44100 , file_path = '',
                                      xscale = 'time' , figsize = (20,4) , fontsize = 20, titles = [],
                                      highlights = np.array([]) , generate_audio_player = True , normalize_audio_player = False,  **kwargs):
    """
    This function plots a time-domain signal x at a sampling
    frequency of sr and highlights regions specified in the
    array highlights.

    ========
     Inputs
    ========
    
    x : (n,) or (n,k) np.ndarray
        A numpy array containing values between -1 and 1.
        The elements represent n samples taken over k
        channels (or 1 channel if the array shape is (n,)).
        Default is a single-channel, one-second 440Hz sine
        tone sampled at 44100Hz.
    sr : int
        The sampling frequency of x. Default is 44100 Hz.
    file_path : str
        If specified, any arguments entered for x and sr will
        be ignored and the signal to plot will be loaded a .wav
        file (using soundfile) specified by file_path.
    xscale : str
        'time' or 'samples'.
        If 'time', then the x-axis will be time in seconds.
        If 'samples', then the x-axis will be the integer index
        of the plotted sample.
    figsize : 2-dimension tuple of int
        Specifies the size of the figure to plot when calling
        plt.figure().
    fontsize : int
        The font size for the title of the plots.
        The font size of other elements in the plot will be 
        resized relative to this number. If plot == False, this
        argument is ignored.
    titles : list of str
        A list of length k (= number of channels in x) to be
        used as the title of the plot of the signal in each
        channel in the order given by x. If the list is empty,
        a default title will be used for each channel.
    highlights : (n,2) or (k,n,2) np.ndarray
        If an (n,2) array, then each row denotes the start
        and end times to be highlighted for every channel in the
        signal x. If a (k,n,2) array, then the ith element of
        the array is a (n,2) array with each row denoting the
        start and end times to be highlighted for channel i.
        If xscale == 'time', then each row must contain the
        start and end times in SECONDS.
        If xscale == 'samples', then each row must contain the
        start and end time INDICES.
        In total, n areas will be highlighted.
    generate_audio_player : bool
        If True, will generate an audio player using the IPython
        kernel.
        If False, no audio player is generated.
        Set to False for long tracks to save processing time.
    normalize_audio_player : bool
        If True, will normalize x to the range [-1,1] before
        generating the audio player with it.
        If False, will not normalize x.
        In either case, the plotted time-domain graph does NOT
        show the normalized x.
    **kwargs : dict
        Extra keyword arguments to control the shading with
        plt.axvspan (e.g. facecolor, alpha, etc.)

    =========
     Outputs
    =========
    
    There are no outputs. However, a plot is generated, and if
    generate_audio_player == True, then an audio player is also
    generated.
    
    ==============
     Dependencies
    ==============

    numpy (as np), matplotlib.pyplot (as plt), IPython,
    soundfile (as sf)
    
    ================
     Future updates
    ================

    Add in **kwargs for toggling plot options (e.g. colour, size, etc.)
    
    """

    ## CHECK IF FILEPATH IS SPECIFIED
    if len(file_path) != 0:
        x, sr = sf.read(file_path)
    
    ## STANDARISE ALL FORMATS TO (n,k) ARRAY
    x = np.expand_dims(x, axis = 1) if len(x.shape) == 1 else x
    n_channels = x.shape[1]
    
    ## MAKE ONE PLOT PER CHANNEL
    for channel in range(n_channels):
        plt.figure(figsize = figsize) # Sets size of plot.
        
        if len(titles) == 0:
            plt.title('Channel {}'.format(channel), fontsize = fontsize)
        elif len(titles) != n_channels:
            print('Error: Number of plot titles does not match number of channels! Program terminating.')
            return
        else:
            plt.title(titles[channel], fontsize = fontsize)

        if xscale == 'time':
            plt.plot(  np.linspace(0, np.size(x[:,channel])/sr, num=np.size(x[:,channel]))  ,  x[:,channel]  , 'k') # Plots time-domain signal in units of seconds.
                                                                                   # The time indices are generated using np.linspace to ensure they have the same dimensions as x.
            plt.xlabel('Time/s', fontsize = fontsize) # Add in x-axis label to graph.
        elif xscale == 'samples':
            plt.plot(x[:,channel], 'k') # Plots time-domain signal in units of samples.
            plt.xlabel('Sample number', fontsize = fontsize)
        else:
            print('Error: Invalid argument entered for xscale. Program terminating.')
            return
        
        if np.any(highlights.shape) and not np.all(highlights.shape):
            print('Error: highlights should either be completely empty or have all nonempty dimensions. Program terminating')
            return
        elif len(highlights.shape) == 2: # i.e. highlights is an (n,2) array (then we have stuff to shade)
            for i in range(highlights.shape[0]):
                plt.axvspan(highlights[i,0],highlights[i,1], **kwargs) # Highlights vertical regions that have been delineated by intervals.
        elif len(highlights.shape) == 3: # i.e. highlights is a (k,n,2) array...
            if highlights.shape[0] != n_channels:
                print('Error: Number of channels in highlights does not match number of channels in x! Program terminating.')
                return
            for i in range(highlights.shape[1]):
                plt.axvspan(highlights[channel,i,0],highlights[channel,i,1], **kwargs) # Highlights vertical regions that have been delineated by intervals.

        plt.xticks(fontsize = 0.7*fontsize) # Set font size for x-axis ticks (i.e. the numbers at each grid line).
        plt.ylabel('Amplitude', fontsize = fontsize) # Add in y-axis label to graph.
        plt.yticks(fontsize = 0.7*fontsize) # Set font size for y-axis ticks (i.e. the numbers at each grid line).
        plt.grid(b=True, which='major', axis='both') # Add in grid lines to graph.

        plt.show() # Last command here displays the actual plot on the IPython console.

        if generate_audio_player == True:
            if np.max(np.abs(x[:,channel])) > 1:
                print('Signal is out of range [-1,1]. Normalising it back to [-1,1] before generating audio player...')
                x = x/np.max(np.abs(x[:,channel]))
            IPython.display.display(IPython.display.Audio(x[:,channel], rate=sr, normalize = normalize_audio_player)) # Actually just IPython.display.Audio(x, rate=sr, normalize = False) will do.
                                                                                                           # We set normalize = False such that we can hear the signal at the exact amplitudes
                                                                                                           # specified in x.
                
#%%==================================================================== read_csv =======================================================================%%#
                
def read_csv(filepath = '', list_of_str = False, delimiter = ',', **kwargs):
    '''
    Reads .csv files and outputs them as a list of
    lists of strings. Sort of a wrapper for
    csv.reader.
    
    ========
     Inputs
    ========
    
    filepath : str
        The path to the .csv file to be read.
        
    list_of_str : bool
        If True, data will be read as though it were a
        list of strings, i.e. as though the .csv file
        contains only one item per row. Otherwise,
        data will be read as though it were 
        a list of lists, i.e. as though the .csv file
        contains more than one item per row.
        It is usually advisable to set this to "True"
        if the .csv file to be read contains only a
        single column.
        
    **kwargs : dict
        Other keyword arguments to pass to the Python
        function open OTHER than 'mode'.
        
    =========
     Outputs
    =========
    
    data : list of lists of str or list of str
        The data read from the .csv file. If
        list_of_str is True, then data[i][j] contains
        the j-th element of the i-th row in the .csv
        file as a string. Otherwise, data[i] contains
        the text in the i-th row in the .csv file.
        
    ==============
     Dependencies
    ==============
    
    csv
    
    =======
     Notes
    =======
    
    Sometimes, depending on the file format, we need
    to add newline = '\n' as an argument in **kwargs
    for the file to be read correctly.
    
    '''
    data = []
    with open(filepath, mode = 'r', **kwargs) as csvfile:
        my_csv_reader = csv.reader(csvfile, dialect = 'excel', delimiter = delimiter)
        if list_of_str:
            for line in my_csv_reader:
                data.append(line[0])
        else:
            for line in my_csv_reader:
                data.append(line)
    return data
                
#%%===================================================================== recorder ======================================================================%%#

def recorder(recording_filename = '', recording_device_index = None, recording_length = 10, recording_sample_frequency = None, buffer_length = 1,
             playback_filename = '', playback_device_index = None, verbose = False):
    """
    Function that allows one to record and/or play back files simultaneously using pyaudio. There are three modes: Playback-only, recording-only, and playback-while-recording.
    The desired mode is determined by whether the playback file name and recording file names are given as empty strings or not.
    
    Playback-only mode is activated if playback_filename is non-empty and recording_filename is empty.
    It will play the audio file specified in playback_filename using tbe device in playback_device_index,
    with the native number of channels and sampling frequency specified in playback_filename.
    
    Recording-only mode is activated if recording_filename is non-empty and playback_filename is empty.
    It will record an audio file of recording_length seconds at recording_sample_frequency to recording_filename using the device in recording_device_index.
    
    Playback-while-recording mode is activated if both playback_filename and recording_filename are non-empty.
    It will record an audio file of recording_length + 2*buffer_length seconds at recording_sample_frequency to recording_filename using the device in recording_device_index.
    Recording will start for buffer_length seconds before playback starts, and will end buffer_length seconds after playback ends.

    ========
     Inputs
    ========

    recording_filename (str): Specifies the name (or filepath) of the .wav file that the recording will be written in.
                              Will create the file and associated directories if it doesn't already exist, and overwrites the file if it already exists.
                              If it is an empty string, then no recording will occur (use this to toggle between recording-only andplayback-only mode).
    recording_device_index (int): Specifies the index of the device that you want to use for recording.
                                  If None, then the default recording (input) device of the system will be used.
                                  See the Help section for more details.
    recording_length (float): The length of the recording in seconds. The default value is 10, giving a recording length of 10 seconds in recording-only mode,
                              and the length of the playback file in playback_filename PLUS 2*buffer_length seconds in recording-while-playback mode.
    recording_sample_frequency (int): The sampling frequency desired for the recording, in Hz. Default value is the default sampling frequency of the device in recording_device_index.
                                      In other words, if recording_device_index == None, then the default value is the default sampling frequency of the default input device.
                                      If manually setting, make sure that this value is compatible with your sound card and microphone/input device, otherwise PyAudio will throw an error.
    buffer_length (float): The length of the buffer in seconds between the start of recording and playback, as well as the end of playback and recording.
                           Is only used in playback-while-recording mode.
    playback_filename (str): Specifies the name (or filepath) of the file that you want to play over the speakers.
                             If it is an empty string, then no playback will occur (you can use this to toggle between recording-only and playback-only mode).
    playback_device_index (int): Specifies the index of the device that you want to use for playback. If None, then the default playback (output) device of the system will be used.
                                 See the Help section for more details.
    verbose (bool): If True, prints out what the turntable is doing and if commands are executed properly. If False, prints out nothing.

    Note that if both playback_filename and recording_filename are empty strings, then this function does nothing.

    =========
     Outputs
    =========

    None
    
    ==============
     Dependencies
    ==============

    os, pyaudio, time, wave

    ======
     Help
    ======

    To obtain playback_device_index and recording_device_index for the desired playback/recording devices, use the following code:

    import pyaudio
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print (i, p.get_device_info_by_index(i).get('name'))

    Also, note that the playback-and-recording mode of this function can also be done in Audacity, if the 'overdub' function is enabled (Edit -> Preferences -> Recording).
    However, in practice, this code is combined with the turntable, so we need to do everything in Python.
    
    Debugging note: The most likely culprit for recording failures is possibly the value of chunk_size, which can be increased manually if necessary.
    """

    p = pyaudio.PyAudio() # Define a new pyaudio object for handling the recording and playback of files.
#%%
    if len(playback_filename) > 0 and len(recording_filename) == 0: # This specifies playback-only mode.
        data_play = wave.open(playback_filename, 'rb') # Open the wave file specified in playback_filename and retrieve its associated metadata. The 'rb' argument refers to read-only mode.

        if verbose:
            print('Playback-only mode selected.')
            print(playback_filename + ' was loaded with {:d} channels at a sampling frequency of {:.0f} Hz.'.format(data_play.getnchannels(),data_play.getframerate()))

            channel_warning_string = 'Warning : Number of output channels ({:d}) of device exceeds number of channels ({:d}) in playback file. This may lead to distortions in playback or errors due to pyaudio.'
            if playback_device_index == None:
                if p.get_default_output_device_info()['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_default_output_device_info()['maxOutputChannels'],data_play.getnchannels()))
            else:
                if p.get_device_info_by_index(playback_device_index)['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_device_info_by_index(playback_device_index)['maxOutputChannels'],data_play.getnchannels()))


        def callback(in_data, frame_count, time_info, flag): # Define pyaudio callback object (this will be activated to load new frames into the buffer when it is empty).
            data = data_play.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # Define a new output stream called playback_stream (similar to MATLAB audiorecorder object).
        playback_stream = p.open(format = p.get_format_from_width(data_play.getsampwidth()),
                                 channels = data_play.getnchannels(), # This is the number of channels of audio in playback_filename.
                                                                      # Must be below the maximum number of output channels allowed for the particular device
                                 rate = data_play.getframerate(), # This is the sampling frequency of audio in playback_filename.
                                 output = True, # Specifies that this stream is an output stream, i.e. the one that we want to use for playback.
                                 stream_callback = callback, # Specifies the callback function to load more frames into the stream when buffer is emptied.
                                 output_device_index = playback_device_index) # Remember to match the output device according to the list in p.get_device_info_by_index if manually changing.

        # Start the playback (i.e. output) stream.
        playback_stream.start_stream() # Starts the playback stream. This line is the one that plays back the audio.
        if verbose:
            print('Playback started.')

        while playback_stream.is_active(): # This loop waits until the stream ends (i.e. until the file finishes playing) before stopping it.
                                           # Callback function is continuously called while the stream is active.
            time.sleep(0.1)

        playback_stream.stop_stream() # Stops the playback (i.e. output) stream.
        if verbose:
            print('Playback ended.')
        playback_stream.close() # Streams must be closed when they are done being used, to prevent errors in future playback.
        data_play.close()
        if verbose:
            print('Playback (output) stream closed.')
        p.terminate() # Close pyaudio.
        if verbose:
            print('PyAudio closed.')
#%%
    elif len(playback_filename) == 0 and len(recording_filename) > 0: # This specifies recording-only mode.
        if verbose:
            print('Recording-only mode selected.')

        if recording_device_index == None: # then we will use the default input device.
            recording_device_index = p.get_default_input_device_info()['index']

        if recording_sample_frequency == None: # then we will use the default sample rate of the device in recording_device_index.
            recording_sample_frequency = int(p.get_device_info_by_index(recording_device_index)['defaultSampleRate']) # The value obtained from the function call is a float, so we need to typecast it to int first.

        frames = [] # Initialise the list of values that we want to store the recorded data in.
        chunk_size = 1 # chunk_size is the number of data points to store in the buffer at a time before writing to frames.
                       # We set it to 1 for lowest latency.
                       # The default value for PyAudio.open() is 1024, and should not cause issues in most systems,
                       # so we have not specified it as a parameter in the recorder function. However, feel free to
                       # change this value within the function for future debugging purposes if there are buffer overflow
                       # problems or samples dropping etc.
        nchunks = int(recording_sample_frequency / chunk_size * recording_length)     # Since we read data in blocks of chunk_size, the number of chunks (nchunks)
                                                                                      # that we need to read is calculated by taking the total number of samples
                                                                                      # divided by chunk_size. The total number of samples to take is, of course,
                                                                                      # recording_sample_frequency * recording_length.
                                                                                      # We could have added 1 to the number of chunks to ensure that the recording is at least
                                                                                      # recording_length seconds long, although that is not done here.
        nchannels = p.get_device_info_by_index(recording_device_index)['maxInputChannels'] # This is the number of channels used for recording. We just use
                                                                                           # the maximum number of input (recording) channels that are present
                                                                                           # in the recording device.
                                                                                           # Can be changed in the future to any number less than that to save memory space too.

        # Define a new input stream called recording_stream (similar to MATLAB audiorecorder object).
        recording_stream = p.open(format = pyaudio.paInt16, # Specifies the recording format to be 16-bit integers
                                  channels = nchannels, 
                                  rate = recording_sample_frequency, # Record at the rate specified by recording_sample_frequency.
                                  input = True, # Specifies that this stream is an input stream, i.e. the one that we want to use for recording.
                                  frames_per_buffer = chunk_size, # Tell PyAudio that we want to record in chunks of chunk_size.
                                  input_device_index = recording_device_index) # Remember to match the input device according to the list in p.get_device_info_by_index if manually changing.

        if verbose:
            print('Recording stream opened with {} channels at a sampling frequency of {} Hz.'.format(nchannels, recording_sample_frequency))
            print('Recording will last for {} seconds.'.format(recording_length))

        try: # We use a try-except statement for the recording part because we don't want to save any recorded data even if there are any errors with the code/recording.
            time_elapsed = 0 # Initialise the time elapsed to 0 (seconds). This value will only be used in verbose mode to print the current recording status.

            # Start the recording.
            for i in range(nchunks): # Read data for nchunks number of times with each block having chunk_size amount of data.
                data = recording_stream.read(chunk_size) # Read (i.e. record) a chunk_size block of data from recording_stream.
                frames.append(data) # Add the read (i.e. recorded) data to the list (frames) that we initialised previously.
                if verbose: # Then print the time elapsed each second.
                    if i * chunk_size  / recording_sample_frequency > time_elapsed + 1: # i*chunk_size/recording_sample_frequency is the total amount of time elapsed so far.
                                                                                        # So if that is greater than time_elapsed + 1, it means that 1 second has elapsed.
                            time_elapsed += 1
                            print('Time elapsed : {} seconds.'.format(time_elapsed) + chr(127) * 10, end = '\r') # We print the time elapsed (rewriting the line each second due
                                                                                                                # to the carriage return \r) every second. The delete characters
                                                                                                                # (denoted by chr(127)) are there to erase any possible
                                                                                                                # extra characters to ensure a clean output.
                                                                                                                # See https://realpython.com/python-print/ for more details on
                                                                                                                # animations with the Python print function.

            recording_stream.stop_stream() # Stops the recording (i.e. input) stream.
            if verbose:
                print('Recording ended.' + chr(127) * 10) # We add the delete characters (chr(127)) again to ensure a clean output.

            recording_stream.close() # Streams must be closed when they are done being used, to prevent errors in future recording.
            if verbose:
                print('Recording (input) stream closed.')
        except KeyboardInterrupt:
            print('Function has been terminated by user. Stopping recording and outputting data acquired until now, if any.')
        except MemoryError:
            print('Memory is full. Stopping recording and outputting data acquired until now, if any.')   
        except: # If any other error occurs...
            raise # raise it (i.e. print the error text)
        finally:
            if verbose:
                print('Writing recorded data to ' + recording_filename)
                
            recording_folder = ''.join([i + os.sep for i in recording_filename.split(os.sep)[:-1]]) # Remove the ~.wav filename part (last element of recording_filename.split)
                                                                                                    # from the path provided to create the directory.
            
            if len(recording_folder) > 0 and not os.path.exists(recording_folder): # If the folder specified in the file path doesn't exist, then create it.
                os.makedirs(recording_folder)
                
            data_rec = wave.open(recording_filename, 'wb') # Open a wave stream to start writing data into the wave file.
            data_rec.setnchannels(nchannels)
            data_rec.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # We want to output in 16-bit float format, so we set the sample width to 2 bytes = 16 bits.
            data_rec.setframerate(recording_sample_frequency)
            data_rec.writeframes(b''.join(frames)) # The letter b before the string indicates byte-type values in frames. Since we read in blocks of chunk_size, the final
                                                   # .wav file that is written will be slightly longer than the desired timing in seconds.
            data_rec.close()
            if verbose:
                print('Finished writing recorded data to ' + recording_filename)

            p.terminate() # Close pyaudio at the very end. Notice the first-in-last-out policy of closing streams here.
            if verbose:
                print('PyAudio closed.')
#%%
    elif len(playback_filename) > 0 and len(recording_filename) > 0: # This specifies playback-while-recording mode.
        if verbose:
            print('Playback-while-recording mode selected.')
            
        # Initialisation of playback parameters 
        data_play = wave.open(playback_filename, 'rb') # Open the wave file specified in playback_filename and retrieve its associated metadata. The 'rb' argument refers to read-only mode.
        if verbose:
            print(playback_filename + ' was loaded with {:d} channels at a sampling frequency of {:.0f} Hz.'.format(data_play.getnchannels(),data_play.getframerate()))

            channel_warning_string = 'Warning : Number of output channels ({:d}) of device exceeds number of channels ({:d}) in playback file. This may lead to distortions in playback or errors due to pyaudio.'
            if playback_device_index == None:
                if p.get_default_output_device_info()['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_default_output_device_info()['maxOutputChannels'],data_play.getnchannels()))
            else:
                if p.get_device_info_by_index(playback_device_index)['maxOutputChannels'] < data_play.getnchannels():
                    print(channel_warning_string.format(p.get_device_info_by_index(playback_device_index)['maxOutputChannels'],data_play.getnchannels()))

        def callback(in_data, frame_count, time_info, flag): # Define pyaudio callback object (this will be activated to load new frames into the buffer when it is empty).
            data = data_play.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # Initialisation of recording parameters.
        if recording_device_index == None: # then we will use the default input device.
            recording_device_index = p.get_default_input_device_info()['index']

        if recording_sample_frequency == None: # then we will use the default sample rate of the device in recording_device_index.
            recording_sample_frequency = int(p.get_device_info_by_index(recording_device_index)['defaultSampleRate']) # The value obtained from the function call is a float,
                                                                                                                      # so we need to typecast it to int first.

        recording_length = data_play.getnframes()/data_play.getframerate() # Set the length of the recording to the length of the playback file.

        frames = [] # Initialise the list of values that we want to store the recorded data in.
        chunk_size = 1 # chunk_size is the number of data points to store in the buffer at a time before writing to frames.
                       # We set it to 1 for lowest latency.
                       # The default value for PyAudio.open() is 1024, and should not cause issues in most systems,
                       # so we have not specified it as a parameter in the recorder function. However, feel free to
                       # change this value within the function for future debugging purposes if there are buffer overflow
                       # problems or samples dropping etc.
        buffer_chunks = int(recording_sample_frequency / chunk_size * buffer_length) # This is the number of blocks to read for the buffer (calculation has same idea as nchunks).
        nchunks = int(recording_sample_frequency / chunk_size * recording_length)     # Since we read data in blocks of chunk_size, the number of chunks (nchunks)
                                                                                      # that we need to read is calculated by taking the total number of samples
                                                                                      # divided by chunk_size. The total number of samples to take is, of course,
                                                                                      # recording_sample_frequency * recording_length.
                                                                                      # We could have added 1 to the number of chunks to ensure that the recording is at least
                                                                                      # recording_length seconds long, although that is not done here.
        nchannels = p.get_device_info_by_index(recording_device_index)['maxInputChannels'] # This is the number of channels used for recording. We just use
                                                                                           # the maximum number of input (recording) channels that are present
                                                                                           # in the recording device.
                                                                                           # Can be changed in the future to any number less than that to save memory space too.

        # Define a new input stream called recording_stream (similar to MATLAB audiorecorder object).
        recording_stream = p.open(format = pyaudio.paInt16, # Specifies the recording format to be 16-bit integers
                                  channels = nchannels, 
                                  rate = recording_sample_frequency, # Record at the rate specified by recording_sample_frequency.
                                  input = True, # Specifies that this stream is an input stream, i.e. the one that we want to use for recording.
                                  frames_per_buffer = chunk_size, # Tell PyAudio that we want to record in chunks of chunk_size.
                                  input_device_index = recording_device_index) # Remember to match the input device according to the list in p.get_device_info_by_index if manually changing.

        if verbose:
            print('Recording stream opened with {} channels at a sampling frequency of {} Hz.'.format(nchannels, recording_sample_frequency))
            print('Buffer of {} seconds will be added to start and end of playback.'.format(buffer_length))

        try: # We use a try-except statement for the recording part because we don't want to save any recorded data even if there are any errors with the code/recording.
            time_elapsed = 0 # Initialise the time elapsed to 0 (seconds). This value will only be used in verbose mode to print the current recording status.

            # Start the recording.
            for i in range(nchunks + 2*buffer_chunks): # First read data for buffer_chunks number of times with each block having chunk_size amount of data.
                data = recording_stream.read(chunk_size) # Read (i.e. record) a chunk_size block of data from recording_stream.
                frames.append(data) # Add the read (i.e. recorded) data to the list (frames) that we initialised previously.
                if verbose: # Then print the time elapsed each second.
                    if i * chunk_size  / recording_sample_frequency > time_elapsed + 1: # i*chunk_size/recording_sample_frequency is the total amount of time elapsed so far.
                                                                                        # So if that is greater than time_elapsed + 1, it means that 1 second has elapsed.
                            time_elapsed += 1
                            print('Time elapsed : {} seconds.'.format(time_elapsed) + chr(127) * 10, end = '\r') # We print the time elapsed (rewriting the line each second due
                                                                                                                # to the carriage return \r) every second. The delete characters
                                                                                                                # (denoted by chr(127)) are there to erase any possible
                                                                                                                # extra characters to ensure a clean output.
                                                                                                                # See https://realpython.com/python-print/ for more details on
                                                                                                                # animations with the Python print function.
                    if i == nchunks + buffer_chunks:
                        print('Playback ended.' + chr(127) * 15)
                if i == buffer_chunks: # Once the first buffer_chunks blocks have been recorded, we start the playback.
                    # Define a new output stream called playback_stream (similar to MATLAB audiorecorder object).
                    playback_stream = p.open(format = p.get_format_from_width(data_play.getsampwidth()),
                                 channels = data_play.getnchannels(), # This is the number of channels of audio in playback_filename.
                                                                      # Must be below the maximum number of output channels allowed for the particular device
                                 rate = data_play.getframerate(), # This is the sampling frequency of audio in playback_filename.
                                 output = True, # Specifies that this stream is an output stream, i.e. the one that we want to use for playback.
                                 stream_callback = callback, # Specifies the callback function to load more frames into the stream when buffer is emptied.
                                 output_device_index = playback_device_index) # Remember to match the output device according to the list in p.get_device_info_by_index if manually changing.
                    if verbose:
                        print('Playback started.' + chr(127) * 10)

            # Start the playback (i.e. output) stream.
            playback_stream.start_stream() # Starts the playback stream. This line is the one that plays back the audio.
            
            playback_stream.close() # Streams must be closed when they are done being used, to prevent errors in future playback.
            data_play.close()
            if verbose:
                print('Playback (output) stream closed.')

            recording_stream.stop_stream() # Stops the recording (i.e. input) stream.
            if verbose:
                print('Recording ended.' + chr(127) * 10) # We add the delete characters (chr(127)) again to ensure a clean output.

            recording_stream.close() # Streams must be closed when they are done being used, to prevent errors in future recording.
            if verbose:
                print('Recording (input) stream closed.')
        except KeyboardInterrupt:
            print('Function has been terminated by user. Stopping recording and outputting data acquired until now, if any.')
        except MemoryError:
            print('Memory is full. Stopping recording and outputting data acquired until now, if any.')   
        except: # If any other error occurs...
            raise # raise it (i.e. print the error text)
        finally:
            if verbose:
                print('Writing recorded data to ' + recording_filename)

            recording_folder = ''.join([i + os.sep for i in recording_filename.split(os.sep)[:-1]]) # Remove the ~.wav filename part (last element of recording_filename.split)
                                                                                                    # from the path provided to create the directory.

            if len(recording_folder) > 0 and not os.path.exists(recording_folder): # If the folder specified in the file path doesn't exist, then create it.
                os.makedirs(recording_folder)

            data_rec = wave.open(recording_filename, 'wb') # Open a wave stream to start writing data into the wave file.
            data_rec.setnchannels(nchannels)
            data_rec.setsampwidth(p.get_sample_size(pyaudio.paInt16)) # We want to output in 16-bit float format, so we set the sample width to 2 bytes = 16 bits.
            data_rec.setframerate(recording_sample_frequency)
            data_rec.writeframes(b''.join(frames)) # The letter b before the string indicates byte-type values in frames. Since we read in blocks of chunk_size, the final
                                                   # .wav file that is written will be slightly longer than the desired timing in seconds.
            data_rec.close()
            if verbose:
                print('Finished writing recorded data to ' + recording_filename)

            p.terminate() # Close pyaudio at the very end. Notice the first-in-last-out policy of closing streams here.
            if verbose:
                print('PyAudio closed.')
#%%
    else: # This means that both playback_filename and recording_filename are empty strings.
        if verbose:
            print('No playback or recording filenames specified. Program terminating.')

#%%=================================================== right_padded_split_track_by_silence =============================================================%%#

def right_padded_split_track_by_silence(input_filename = None, x = [],
                                        sr = None, output_directory = '', split_length = 4, top_db = 60, tolerance_ratio = 0, verbose = False):
    """
    Function that splits track by identifying silent segments and outputs non-silent segments of split_length with at least tolerance_ratio of non-silence
    to silence. Segments are preferentially right-padded with zeroes.
    
    In other words, the method used is:

    1. Initialise an empty buffer B of split_length seconds to store any output data. 
    2. Given an n-by-2 matrix intervals from librosa.effects.split, split the data into intervals labelled "silence" and "non-silence" alternatingly.
       There should be n non-silent intervals and up to n+1 silent intervals (depending on whether the start and end of the track is considered non-silent.
       For the purposes of this algorithm, we assume n non-silent intervals and n+1 silent intervals for a total of 2n+1 intervals.
    3. For each interval I_k, where k iterates through {1, 2, ..., 2n+1} (i.e. backward in time) starting from the latest non-silent interval,
        1. If I_k is silent, keep adding samples from I_k until either:
            1. The buffer B is full: Then discard all remaining samples from I_k, output the buffer, and move to I_{k+1}.
            2. There are no more samples to add to B: Then move to I_{k+1}.
        2. If I_k is non-silent, keep adding samples from I_k until either:
            1. The buffer B is full: Then output the buffer, and continue adding samples from I_k into B.
               Repeat this action until there are no more samples to add to B.
            2. There are no more samples to add to B: Then move to I_{k+1}.
    4. At the end, right-pad the buffer with silence and output it.
    5. Output the buffer if there is at least tolerance_ratio of non-silence to silence, else discard it.

    If a file name is specified, then the .wav file specified by that file name will be used for splitting.
    If a file name is not specified, then a nonempty numpy.ndarray of values between -1 and +1 must be specified instead.
    If no file name or nonempty numpy.ndarray is specified, then a default numpy.ndarray of 2 sets of alternating 2s white
    noise and 2s silence sampled at 44100Hz will be used as the signal.
    If both a file name and nonempty numpy.ndarray is specified, then the file name will take precedence. In other words,
    the numpy.ndarray will be ignored and the .wav file specified by the file name will be used for splitting.

    ========
     Inputs
    ========

    input_filename : str or None
        The name or path of the input file in .wav format.
        If None, then it is assumed that a numpy.ndarray is used for
        input.
        The file must contain an input signal of floating point
        numbers between -1 and +1.
    x : 1-dimensional numpy.ndarray or None
        An input signal of floating point numbers between -1 and +1.
        If None, then it is assumed that a file name is used for
        input.
        Default signal is 2 sets of alternating 1s white noise and 
        1s silence sampled at 44100Hz.
    sr : int or None
        The sampling rate of the output files that are written.
        If None, then this will be set to the sampling rate of the
        file in input_filename if it is nonempty, and 44100 if
        input_filename is nonempty.
    output_directory : str
        The output directory where the processed files will be 
        written to. If an empty string is provided, then the files
        will be outputted to the current working directory.
        Hence, the default output directory IS the current working
        directory.
    split_length : float
        The length in seconds that each output file should be.
    top_db : float
        The number of dB below a reference level of 1 that a segment
        will be considered as silence. It is exactly the same as the
        top_db parameter in librosa.effects.split.
    tolerance_ratio : float (in [0,1])
        The minimum proportion of non-silent samples in each output
        file.
    verbose : bool
        If true, prints out names of files as they are outputted
        from the buffer.
        If false, nothing is printed out.
    
    =========
     Outputs
    =========

    This function has no outputs. However, it will output files 
    of the format <input_filename>_<output_file_index>.wav to the
    directory specified in output_directory.
    
    ==============
     Dependencies
    ==============
    
    librosa, numpy (as np), os, soundfile (as sf)

    ==========
     Examples
    ==========

    >>> right_padded_split_track_by_silence(output_directory = "sample_track" + os.sep) # Outputs two files, both containing 2s of white noise followed by 2s of silence.

    ================
     Future updates
    ================
    
    None planned at the moment.
    
    """
    #%%== EXCEPTION HANDLING ==%%#
    if input_filename == None:
        input_filename = '' # ...we assign an empty string to input_filename to prevent errors when writing with sf.write later,...
        if sr == None:
            sr = 44100
        if len(x) == 0:
            x = np.concatenate((2*np.random.rand(88200)-1,np.zeros(88200),2*np.random.rand(88200)-1,np.zeros(88200))) # ...and set a default signal of alternating white noise and silence.
    else: 
        x, sr = librosa.load(input_filename, sr = sr, mono = True, dtype = np.float32) # Recall that librosa.load outputs both the signal and its sampling frequency.
   
    if len(output_directory) > 0 and (not os.path.exists(output_directory)): # If the output directory is not an empty string and doesn't yet exist,...
        os.makedirs(output_directory) # ...then we create it (as an empty directory of course).

    # Use librosa.effects.split to obtain non-silent intervals in x.
    intervals = librosa.effects.split(x, top_db = top_db, ref = 1, frame_length=2048, hop_length=512) # Is a 2-column array of start and end time indices.
                                                                                                      # Start indices are inclusive and end indices are exclusive,
                                                                                                       # as per regular Python syntax.
    if len(intervals) == 0: # If the entire track is silent,...
        print('Track is silent. Nothing to output. Program terminating.')
        return # ...then there's nothing to write.

    if int(split_length*sr) < 1: # If the desired buffer size is less than the time between each sample,...
        print('Split lengths are too small for sampling frequency. Increase either quantity and try again. Program terminating.')
        return # ...then it's impossible to output anything and there's nothing to write.
    
    if tolerance_ratio != np.min([np.max([tolerance_ratio,0]),1]): # If the tolerance ratio is outside of [0,1],...
        tolerance_ratio = np.min([np.max([tolerance_ratio,0]),1]) # ...then set it to the bound that is closer to the provided value.
        print('Tolerance ratio out of bounds. Setting it to {:d}.'.format(tolerance_ratio))
        
    #%%== MAIN CODE STARTS HERE ==%%#
    # If the code makes it this far, then intervals is nonempty and there is at least one nonsilent segment in the signal.
    # The 'for' loop marks each element of the signal as silent (False) or non-silent (True).
    sample_is_nonsilent = np.full(len(x),False) # Initialisation.
    for i in range(intervals.shape[0]):
        sample_is_nonsilent[intervals[i,0]:intervals[i,1]] = np.full(intervals[i,1]-intervals[i,0],True)
        
    buffer = np.zeros(int(split_length*sr)) # Initialise empty buffer of split_length seconds.
                                            # If the code makes it this far, then buffer is nonempty and is at least one sample long.
                                            # We typecast to int in case split_length*sr is not an integer.
    buffer_position = 0 # Initialise buffer position at the leftmost (first) index.
    buffer_nonsilent_samples = 0 # Initialise variable to track numbers of non-silent samples in buffer.
    output_file_index = 1 # This is the current index of the file that will be output next from the buffer. Initialised to 1.
    output_filename = (output_directory # Start with the specified output directory, then follow with the filename
                      + input_filename.split(os.sep)[-1][:-4] # Remove the last four characters of the input filename because it is '.wav'. We use split on os.sep in case a path was provided.
                      + '_{:04d}.wav' # Add a placeholder for the counter to the filename
                      if len(input_filename) > 1 else 
                      output_directory # Start with the specified output directory, then follow with the filename
                      + '{:04d}.wav') # Add a placeholder for the counter to the filename

    for curr_signal_position in range(len(x)): # Iterate backwards through all signal positions.
        # Test if the buffer is empty or full (it's possible that the buffer is neither empty nor full --- when it's partially filled)
        buffer_is_empty = True if buffer_position == 0 else False
        buffer_is_full = True if buffer_position >= len(buffer) else False

        if buffer_is_empty: # (1) If the buffer is empty,...
            if sample_is_nonsilent[curr_signal_position]: # (1A) ...and if sample is nonsilent at the current signal position,...
                buffer[buffer_position] = x[curr_signal_position] # ...then add the current sample to the buffer,...
                buffer_nonsilent_samples += 1 # ...increment the counter for nonsilent samples in the buffer,...
                buffer_position += 1 # ...and move the buffer position by one unit to the right.
            else: # (1B) ...and if the the sample if silent at the current signal position,...
                continue # ...then do nothing (i.e. we discard the current sample).
        elif buffer_is_full: # (2) If the buffer is full,...
            if sample_is_nonsilent[curr_signal_position]: # (2A) ...and if sample is nonsilent at the current signal position,...
                # ...then we output the buffer (if it is at least tolerance_ratio nonsilent),...
                if buffer_nonsilent_samples/len(buffer) >= tolerance_ratio:
                    sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                             data = buffer, samplerate = sr)
                    if verbose:
                        print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/len(buffer), output_filename.format(output_file_index)))
                    output_file_index += 1 # Increment the output file index for the next file to be outputted from the buffer.
                elif verbose:
                    print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/len(buffer)))
                buffer = np.zeros(int(split_length*sr)) # ...reset the buffer to zero to restart adding samples to the buffer,...
                buffer_position = 0 # ...reset the buffer position to the beginning,...
                buffer_nonsilent_samples = 0 # ...reset the counter for nonsilent samples in the buffer,...
                buffer[buffer_position] = x[curr_signal_position] # ...add the current sample to the buffer,...
                buffer_position += 1 # ...and move the buffer position by one unit to the right.
                # Note that while these steps may seem convoluted, we need the code to work with any buffer size, including a buffer of size 1.
            else: # (2B) ...and if the the sample if silent at the current signal position,...
                continue # ...then do nothing (i.e. we discard the current sample).
        else: # (3) If the buffer is partially filled,...
            buffer[buffer_position] = x[curr_signal_position] # ...then add the current sample to the buffer (regardless of whether it's silent),...
            if sample_is_nonsilent[curr_signal_position]:
                buffer_nonsilent_samples += 1 # ...increment the counter for nonsilent samples in the buffer if the current sample is nonsilent,...
            buffer_position += 1 # ...and move the buffer position by one unit to the right.

    # Output any remaining samples in the buffer (if it is at least tolerance_ratio nonsilent).
    if buffer_nonsilent_samples/len(buffer) >= tolerance_ratio:
        sf.write(file = output_filename.format(output_file_index), # Add the counter to the output filename.
                 data = buffer, samplerate = sr)
        if verbose:
            print('Buffer is {:.1f}% non-silent. Outputting buffer to {}'.format(100*buffer_nonsilent_samples/len(buffer), output_filename.format(output_file_index)))
    elif verbose:
        print('Buffer is {:.1f}% non-silent. Dropping samples in buffer.'.format(100*buffer_nonsilent_samples/len(buffer)))
        
#%%========================================================== split_track_by_silence ===================================================================%%#

def split_track_by_silence(input_filename = None, x = [],
                           sr = None, output_directory = '', split_length = 4, top_db = 60, tolerance_ratio = 0, padding = 'center', verbose = False):
    """
    Function that splits track by identifying silent segments and outputs non-silent segments of split_length with at least tolerance_ratio of non-silence
    to silence. This is a wrapper function for the helper functions left_padded_split_track_by_silence, right_padded_split_track_by_silence, and
    center_padded_split_track_by_silence. Refer to the docstrings of the helper functions for more details on the algorithm.
    
    If a file name is specified, then the .wav file specified by that file name will be used for splitting.
    If a file name is not specified, then a nonempty numpy.ndarray of values between -1 and +1 must be specified instead.
    If no file name or nonempty numpy.ndarray is specified, then a default numpy.ndarray of 2 sets of alternating 2s white
    noise and 2s silence sampled at 44100Hz will be used as the signal.
    If both a file name and nonempty numpy.ndarray is specified, then the file name will take precedence. In other words,
    the numpy.ndarray will be ignored and the .wav file specified by the file name will be used for splitting.

    ========
     Inputs
    ========

    input_filename : str or None
        The name or path of the input file in .wav format.
        If None, then it is assumed that a numpy.ndarray is used for
        input.
        The file must contain an input signal of floating point
        numbers between -1 and +1.
    x : 1-dimensional numpy.ndarray or None
        An input signal of floating point numbers between -1 and +1.
        If None, then it is assumed that a file name is used for
        input.
        Default signal is 2 sets of alternating 1s white noise and 
        1s silence sampled at 44100Hz.
    sr : int or None
        The sampling rate of the output files that are written.
        If None, then this will be set to the sampling rate of the
        file in input_filename if it is nonempty, and 44100 if
        input_filename is nonempty.
    output_directory : str
        The output directory where the processed files will be 
        written to. If an empty string is provided, then the files
        will be outputted to the current working directory.
        Hence, the default output directory IS the current working
        directory.
    split_length : float
        The length in seconds that each output file should be.
    top_db : float
        The number of dB below a reference level of 1 that a segment
        will be considered as silence. It is exactly the same as the
        top_db parameter in librosa.effects.split.
    tolerance_ratio : float (in [0,1])
        The minimum proportion of non-silent samples in each output
        file.
    padding : str ('left' , 'right', or 'center')
        The padding that is used for the output tracks.
    verbose : bool
        If true, prints out names of files as they are outputted
        from the buffer.
        If false, nothing is printed out.
    
    =========
     Outputs
    =========

    This function has no outputs. However, it will output files 
    of the format <input_filename>_<output_file_index>.wav to the
    directory specified in output_directory.
    
    ==============
     Dependencies
    ==============
    
    librosa, numpy (as np), os, soundfile (as sf)

    ==========
     Examples
    ==========

    >>> split_track_by_silence(output_directory = "sample_track" + os.sep, tolerance_ratio = 0.1) # Outputs two files, both containing 2s of white noise center-padded with silence.

    ================
     Future updates
    ================
    
    None planned at the moment.
    
    """
    if padding == 'center':
        center_padded_split_track_by_silence(input_filename = input_filename, x = x, sr = sr, output_directory = output_directory,
                                             split_length = split_length, top_db = top_db, tolerance_ratio = tolerance_ratio, verbose = verbose)
    elif padding == 'left':
        left_padded_split_track_by_silence(input_filename = input_filename, x = x, sr = sr, output_directory = output_directory,
                                           split_length = split_length, top_db = top_db, tolerance_ratio = tolerance_ratio, verbose = verbose)
    elif padding == 'right':
        right_padded_split_track_by_silence(input_filename = input_filename, x = x, sr = sr, output_directory = output_directory,
                                            split_length = split_length, top_db = top_db, tolerance_ratio = tolerance_ratio, verbose = verbose)
    else:
        print('Invalid argument entered for padding. Padding must be either "left", "right", or "center". Function terminating.')
        return
    
#%%================================================================ write_csv ==========================================================================%%#

def write_csv(data, filepath = '', delimiter = ',', list_of_str = False, mode = 'w', **kwargs):
    '''
    Writes a list of lists to a .csv file. Sort of a
    wrapper for csv.writer.
    
    ========
     Inputs
    ========
    
    data : list of lists or list of str
        The data to be written to the .csv file. If a
        list of lists, then data[i][j] should contain
        either a string or an object that can be
        typecasted to a string, and will be written as
        the j-th element of the i-th row in the .csv
        file. If a list of strings, then data[i]
        should contain the string to be written to the
        i-th row of the .csv file.
        
    filepath : str
        The path to write the .csv file to.
        
    delimiter : char
        The delimiter used to separate entries in the
        same row in the .csv file.
    
    list_of_str : bool
        If True, data will be assumed to be a list of
        strings. Otherwise, data will be assumed to be
        a list of lists.
    
    mode : str
        If 'w', will overwrite the file at filepath if
        it already exists. If 'x', will throw an error
        if the file at filepath already exists.
    
    **kwargs : dict
        Other keyword arguments to pass to the Python
        function open OTHER than 'mode'.
    
    =========
     Outputs
    =========
    
    None. However, a .csv file will be written to the
    specified filepath.
    
    ==============
     Dependencies
    ==============
    
    csv
    
    ==========
     Examples
    ==========
    
    # Write a list of strings to a .csv file (Windows)
    >>> my_list = ['alpha','bravo','charlie','delta']
    >>> write_csv(my_list, filepath = 'my_list.csv', list_of_str = True, newline = '\n')
    '''
    with open(filepath, mode = mode, **kwargs) as csvfile: # Save the metadata using csv.writer.
        my_csv_writer = csv.writer(csvfile, delimiter = delimiter, dialect='excel')
        if list_of_str:
            for row in data:
                my_csv_writer.writerow([row])
        else:
            my_csv_writer.writerows(data)