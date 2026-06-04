
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrogram(I_data, Q_data, PRF, time_window_length=128, overlap_factor=0.95, pad_factor=16):
    """
    Generate a spectrogram.

    :param I_data: In-phase radar data (1D or 2D NumPy array).
    :param Q_data: Quadrature radar data (1D or 2D NumPy array).
    :param PRF:    The sampling rate.
    :param time_window_length: Window length (128).
    :param overlap_factor:     Overlap factor (0.95).
    :param pad_factor:         Zero-padding factor (16).

    :return: None (displays the spectrogram plot).
    """
    radar_signal = I_data + 1j * Q_data

    # 2) Compute window/overlap/nfft the same as MATLAB
    overlap_length = int(time_window_length * overlap_factor)  
    nfft = pad_factor * time_window_length  

    f, t, Sxx = spectrogram(
        radar_signal,
        fs=PRF,
        window='hann',           
        nperseg=time_window_length,
        noverlap=overlap_length,
        nfft=nfft,
        return_onesided=False   
    )

   
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)

    f_step = PRF / nfft
   
    f_shifted = np.linspace(-PRF/2, PRF/2 - f_step, nfft)


    Sxx_mag = np.abs(Sxx_shifted)
    Sxx_norm = Sxx_mag / np.max(Sxx_mag)
    Sxx_db = 20 * np.log10(Sxx_norm + 1e-12)  # +1e-12 to avoid log(0)


    plt.figure #(figsize=(6,4))
    plt.imshow(
        Sxx_db, 
        extent=[t[0], t[-1], f_shifted[0], f_shifted[-1]],
        aspect='auto', 
        cmap='jet',
        origin='lower'  # or 'upper' if you want frequency axis reversed
    )
    plt.clim([-60, 0])
    plt.ylim([-20, 20])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.show()


