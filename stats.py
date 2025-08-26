import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pyloudnorm as pyln # rango dinámcico LUFS


path = "media/"


def detect_noise_section(audio, sr, window_duration=0.1, threshold_percentile=5):
    """
    Detects the quietest noise section in an audio file.

    Parameters:
    - audio (numpy.ndarray): Loaded audio signal.
    - sr (int): Sampling rate of the audio.
    - window_duration (float): Duration of the windows in seconds (default: 0.1 s).
    - threshold_percentile (int): Percentile to define the noise threshold (default: 5%).

    Returns:
    - noise_section (numpy.ndarray): Audio signal corresponding to the noise section.
    """
    try:
        # Length of each window in samples
        window_length = int(window_duration * sr)

        # Split audio into windows
        num_windows = len(audio) // window_length
        windows = audio[:num_windows * window_length].reshape(-1, window_length)

        # Calculate RMS for each window
        rms_values = np.sqrt(np.mean(windows**2, axis=1))

        # Determine the threshold based on the percentile
        threshold = np.percentile(rms_values, threshold_percentile)

        # Find the first window that meets the threshold
        for i, rms in enumerate(rms_values):
            if rms <= threshold:
                start_sample = i * window_length
                end_sample = start_sample + window_length
                return audio[start_sample:end_sample]

        print("No region found below the threshold.")

    except Exception as e:
        print(f"Error detecting noise section: {e}")
        return None


def noise_floor(audio, sr, window_duration=0.1, threshold_percentile=5):
    """
    Automatically calculates the noise floor by detecting the quietest section.

    Parameters:
    - audio_path (str): Path to the audio file.
    - window_duration (float): Duration of the windows in seconds (default: 0.1 s).
    - threshold_percentile (int): Percentile to define the noise threshold (default: 5%).

    Returns:
    - noise_floor_db (float): Noise floor level in decibels (dB).
    """
    # Detect the noise section
    noise_section = detect_noise_section(audio, sr, window_duration,threshold_percentile)

    # Calculate the RMS level of the noise
    rms_noise = np.sqrt(np.mean(noise_section**2))

    # Avoid logarithm errors if RMS is zero
    if rms_noise == 0:
        print("The RMS level of the noise is zero. Check the selected section.")
        return None

    # Convert to decibels
    noise_floor_db = 20 * np.log10(rms_noise)
    return noise_floor_db





def dynamic_range1(audio):
    """
    Calculates the dynamic range of an audio file.

    Parameters:
    - audio (numpy.ndarray): Loaded audio signal.

    Returns:
    - dynamic_range_db (float): Dynamic range in decibels (dB).
    """
    try:
        # Calculate the maximum and RMS level of the signal
        max_amplitude = np.max(np.abs(audio))
        rms_amplitude = np.sqrt(np.mean(audio**2))

        # Avoid logarithm errors if RMS is zero
        if rms_amplitude == 0:
            print("The level RMS of the audio is zero. Check the audio file.")
            return

        # Calculate the dynamic range in dB
        dynamic_range_db = 20 * np.log10(max_amplitude / rms_amplitude)

        return dynamic_range_db

    except Exception as e:
        print(f"Error calculating dynamic range: {e}")
        return None




def dynamic_range2(audio, sr, frame_size=2048, hop_size=1024, threshold_db=-60):
    # Absolute peak
    peak = np.max(np.abs(audio))

    # Block division
    n_frames = (len(audio) - frame_size) // hop_size + 1
    rms_values = []
    
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Ignore silence (below threshold)
        if rms_db > threshold_db:
            rms_values.append(rms)
    
    if not rms_values:
        raise ValueError("No useful signal detected above the threshold.")

    # Estimated background noise level (minimum useful RMS)
    noise_floor = min(rms_values)

    # Calculate dynamic range
    dynamic_range = 20 * np.log10(peak / noise_floor)
    return dynamic_range

    ## Convert to mono if stereo
    #if audio.ndim == 2:
    #    audio = np.mean(audio, axis=1)

    dr = calcular_rango_dinamico(audio, sr)
    print(f"Estimated dynamic range: {dr:.2f} dB")





def dynamic_range3(audio, sr):

    # Create EBU R128 meter
    meter = pyln.Meter(sr)  # by default: gating on

    # Perceptual measurements
    loudness = meter.integrated_loudness(audio)
    return loudness






def frequency_range(audio, sr, energy_threshold=1e-06):
    """
    Calculates the minimum and maximum frequency of an audio file with higher precision.

    Parameters:
    - audio (numpy.ndarray): Loaded audio signal.
    - sr (int): Sampling rate of the audio.
    - energy_threshold (float): Minimum energy threshold to consider a frequency (default: 1e-6).

    Returns:
    - min_freq (float): Minimum frequency in Hz.
    - max_freq (float): Maximum frequency in Hz.
    """
    try:
        # Calculate the spectrogram using the Fourier Transform
        stft = np.abs(librosa.stft(audio))

        # Calculate the frequency corresponding to each bin of the spectrogram
        freqs = librosa.fft_frequencies(sr=sr)

        # Sum the energy over time for each frequency
        energy_per_freq = np.sum(stft, axis=1)

        # Filter frequencies by the energy threshold
        significant_indices = np.where(energy_per_freq > energy_threshold)[0]

        if len(significant_indices) == 0:
            print("No significant frequencies found above the threshold.")
            return None, None

        # Determine the minimum and maximum significant frequencies
        min_freq = freqs[significant_indices[0]]
        max_freq = freqs[significant_indices[-1]]

        return min_freq, max_freq

    except Exception as e:
        print(f"Error calculating frequency range: {e}")
        return None, None






def goniometer(file, window_size=128):
    try:        
        data, sr = sf.read(file)    
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None


    # si es mono devolvemos 0
    if len(data.shape)<2:
        return 0.0,0.0

    pi_4 = np.pi/4
    frame = 0

    maxVal, sum, count = 0.0, 0.0, 0

    chunk = data[frame*window_size:(frame+1)*window_size]        
    while chunk.size != 0:
        l, r = chunk[:, 0], chunk[:, 1]   
        
        rad = np.sqrt(np.square(l)+np.square(r))

        theta = (np.arctan2(r,l)+2*np.pi+np.pi/4)%(2*np.pi)
        theta = theta % (np.pi)
        theta = np.abs(theta -np.pi/2)
        theta = theta*rad

        val = np.mean(theta)
        if val>maxVal: 
            maxVal=val
        sum += val
        count += 1

        frame += 1
        if (frame+1)*window_size>=data.size:
            chunk = data[frame*window_size:]        
        else:        
            chunk = data[frame*window_size:(frame+1)*window_size]  
    
    return maxVal, sum/count if count>0 else 0







if __name__ == "__main__":    
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    for f in files:
        # Load audio file
        if f[-3:] == 'm4a':
            os.system(f'ffmpeg -i "{f}" "{f[:-3]}wav"')
            f = f[:-3] + 'wav'

        try:
            audio, sr = sf.read(f)

        except Exception as e:
            print(f"Error loading audio file: {e}")
            continue
            audio = None

        print(f"File: {f}")
        noiseF = noise_floor(audio,sr)
        if noiseF is not None:
            print(f"Noise floor: {noiseF:.2f} dB")


        # Convert to mono if stereo
        if audio.ndim == 2:
            audioM = np.mean(audio, axis=1)
        else:
            audioM = audio

        dynamicR = dynamic_range1(audioM)
        if dynamicR is not None:
            print(f"Dynamic range: {dynamicR:.2f} dB")     

        dynamicR2 = dynamic_range2(audioM,sr)
        if dynamicR2 is not None:
            print(f"Dynamic range 2: {dynamicR2:.2f} dB")     


        loudness = dynamic_range3(audioM,sr)
        print(f"Dynamic range 3 (loudness (LUFS)): {loudness:.2f}")



        min_freq, max_freq = frequency_range(audio, sr, energy_threshold=1000)
        if min_freq is not None and max_freq is not None:
            print(f"Frequency range: {min_freq:.2f} Hz - {max_freq:.2f} Hz")  


        max, avg = goniometer(f)            
        print(f"Stereo width: max={max:.2f}, avg={avg:.2f}")

        print("\n\n")         
    
