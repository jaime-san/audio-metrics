import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pyloudnorm as pyln # rango dinámcico LUFS
from scipy.signal import butter, filtfilt



path = "audios/"

# band of interest (Hz)
f1, f2 = 4000, 6000



def band_energy_fft(audio, sr, fmin, fmax):
    # Compute FFT
    N = len(audio)
    fft_vals = np.fft.rfft(audio)
    fft_freqs = np.fft.rfftfreq(N, 1/sr)
    
    # Power spectrum
    power = np.abs(fft_vals)**2
    
    # Indices within the desired band
    band_idx = np.where((fft_freqs >= fmin) & (fft_freqs <= fmax))
    
    # Energy in the band
    energy = np.sum(power[band_idx])
    return energy



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def band_energy_filtered(audio, sr, fmin, fmax):
    # Design bandpass filter
    b, a = butter_bandpass(fmin, fmax, sr, order=6)
    
    # Apply filter
    filtered = filtfilt(b, a, audio)
    
    # Compute total energy and RMS energy
    energy = np.sum(filtered**2)              # total energy
    rms_energy = np.sqrt(np.mean(filtered**2))  # RMS energy
    return energy, rms_energy





if __name__ == "__main__":    
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    for f in files:
        try:
            audio, sr = sf.read(f)

        except Exception as e:
            print(f"Error loading audio file: {e}")
            continue
            audio = None

        print(f"File: {f}")

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        e1 = band_energy_fft(audio, sr, f1, f2)  # energy between f1-f2 Hz
        print(f"FFT method Energy in the band {f1}-{f2} Hz: {e1:.2f}")


        e2, e2_rms = band_energy_filtered(audio, sr, f1, f2)
        print(f"Filtering method. Energy in the band {f1}-{f2} Hz: {e2:.2f}")
        print(f"                  RMS energy in the band {f1}-{f2} Hz: {e2_rms:.4f}")

        print("")

