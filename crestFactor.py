#%%
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import pyloudnorm as pyln # rango dinámcico LUFS

path = "audios/"

# seconds of audio to process
# secs=0 means process the entire file
secs = 10



def crest_factor(signal, frame_size=1024, hop_length=512):
    """
    Computes the crest factor of a signal using a sliding window.

    Args:
        signal (array): The input signal.
        frame_size (int): The size of each frame in samples.
        hop_length (int): The number of samples between consecutive frames.

    Returns:
        np.array: An array of crest factor values.
    """
    res = []
    for i in range(0, len(signal), hop_length):
        # Get a portion of the signal
        cur_portion = signal[i:i + frame_size]  
        # Compute the RMS energy for the portion
        rmse_val = np.sqrt(1 / len(cur_portion) * sum(i ** 2 for i in cur_portion))  
        # Compute the crest factor
        crest_val = max(np.abs(cur_portion)) / rmse_val  

        # convert to dB
        if crest_val > 0:
            crest_val = 20 * np.log10(crest_val)  
        else:
            crest_val = 0
        # Store the crest factor value
        res.append(crest_val)  
    # Convert the result to a NumPy array
    return np.array(res)  

def plot_crest_factor(signal, sr, name, frame_size=1024, hop_length=512):
    """
    Plots the crest factor of a signal over time.

    Args:
        signal (array): The input signal.
        name (str): The name of the signal for the plot title.
        frame_size (int): The size of each frame in samples.
        hop_length (int): The number of samples between consecutive frames.
    """
    # Compute the crest factor
    crest = crest_factor(signal, frame_size, hop_length)  
    # Generate the frame indices
    frames = range(0, len(crest))  
    # Convert frames to time
    time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)  
    # Create a new figure with a specific size
    plt.figure(figsize=(15, 7))  
    # Plot the crest factor over time
    plt.plot(time, crest, color="b")  
    # Set the title of the plot
    #plt.title(name + " (Crest Factor)")  
    # Show the plot
    #plt.show()  
    plt.xlabel("Time (seconds)")
    plt.ylabel("dB")
    plt.savefig(name[:-3] + "png")


def mean_max_crest_factor(signal, frame_size=1024, hop_length=512):
    """
    Computes the mean and the max crest factor of a signal.

    Args:
        signal (array): The input signal.
        frame_size (int): The size of each frame in samples.
        hop_length (int): The number of samples between consecutive frames.

    Returns:
        float: The mean crest factor value.
    """
    crest = crest_factor(signal, frame_size, hop_length)  
    return np.mean(crest), np.max(crest)



if __name__ == "__main__":    
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    for f in files:
        try:
            audio, sr = sf.read(f)

        except Exception as e:
            print(f"Error loading audio file: {e}")
            continue
            audio = None

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # select initial portion of audio
        if secs>0:
            audio = audio[:secs*sr]

        print(f"File: {f}")

        plot_crest_factor(audio, sr, f)        
        
        mean_crest, max_crest = mean_max_crest_factor(audio)
        print(f"Mean Crest Factor: {mean_crest:.2f} dB    Max Crest Factor {max_crest:.2f} dB")
        print()


