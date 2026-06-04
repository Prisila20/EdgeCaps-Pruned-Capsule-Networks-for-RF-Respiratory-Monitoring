#!/usr/bin/env python3
import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import butter, filtfilt, spectrogram
from plot_spectrogram import generate_spectrogram

# New imports for logging and system monitoring
import psutil
import csv
from datetime import datetime

# Optional: Power monitoring via INA219 sensor
try:
    from adafruit_ina219 import INA219
    import board
    import busio

    i2c_bus = busio.I2C(board.SCL, board.SDA)
    ina = INA219(i2c_bus)
    power_monitoring = True
except Exception as e:
    print("INA219 sensor not found or failed to initialize.")
    power_monitoring = False

# CSV log file setup for recording metrics
LOG_FILE = "inference_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "BatchID", "Prediction", "Latency(ms)",
            "CPU(%)", "RAM(%)", "Temp(C)", "Power(mW)"
        ])

def load_shared_data(shared_folder="/shared_data"):
    """
    Load the current batch data and its metadata from the shared folder.
    Returns:
      - batch_data: dictionary with keys "i_data" and "q_data"
      - metadata: dictionary containing batch info (e.g., batch_id, timestamp)
    """
    current_file = os.path.join(shared_folder, "current.npy")
    metadata_file = os.path.join(shared_folder, "batch_info.json")
    
    if not os.path.exists(current_file) or not os.path.exists(metadata_file):
        return None, None
    
    batch_data = np.load(current_file, allow_pickle=True).item()
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return batch_data, metadata

def filter_breathing_signal(signal, fs=17.0):
    """
    Apply a 2nd-order Butterworth bandpass filter to extract the low-frequency breathing signal.
    Cutoff frequencies: 0.1 Hz to 0.6 Hz.
    """
    low_cut = 0.1
    high_cut = 0.7
    nyquist = fs / 2.0
    b, a = butter(2, [low_cut / nyquist, high_cut / nyquist], btype='bandpass')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def preprocess_spectrogram(I_data, Q_data, fs=17.0, save_path="spectrogram.png"):
    """
    Generate a spectrogram image from I/Q data, save it to a file,
    and convert it into a normalized tensor for model input.
    """
    plt.figure(figsize=(6, 4))
    generate_spectrogram(I_data, Q_data, fs)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved to: {save_path}")

    # Load and preprocess the image for the model
    img = Image.open(save_path).convert('RGB')
    img = img.resize((64, 64))  # Adjust to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def classify_spectrogram(img_tensor, model):
    """
    Run inference on the spectrogram tensor using the loaded model and return the prediction.
    """
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output, dim=1)
        # Adjust class names to match your model's output
        class_names = ["Bradypnea", "Tachypnea", "Eupnea"]
        prediction = class_names[pred_idx.item()]
    return prediction

def process_batch(batch_data, PRF=500.0, save_path="spectrogram.png"):
    """
    Process batched I/Q data. Reconstructs the range axis, selects the desired bins,
    combines I and Q into a complex signal, removes DC, and applies separate filtering
    to the real and imaginary parts.
    Returns the filtered I and Q signals.
    """
    # Retrieve batched I and Q data (assumed shape: [num_frames, num_bins])
    i_data = batch_data["i_data"]
    q_data = batch_data["q_data"]

    frame_start, bin_len, frame_stop = 0.1858, 8 * 1.5e8 / 23.328e9 , 2.5000
    range_vector = np.arange(frame_start - 1e-5, frame_stop + 1e-5, bin_len)
    


    range_idx = np.where((range_vector >= 0.10858) & (range_vector <= 1.1100))[0]
    print("Selected range indices:", range_idx)

    # Form a complex IQ signal from batched I and Q data:
    Data_IQ = i_data + 1j * q_data

    # Select only the desired range bins:
    sig = Data_IQ[:, range_idx]
    print("Selected signal shape:", sig.shape)

    # Average across the selected range bins to obtain a 1D slow-time signal per frame:
    slow_time_signal = np.mean(sig, axis=1)
    slow_time_signal_no_dc = slow_time_signal - np.mean(slow_time_signal)

    sig -= sig.mean(axis=0)
    b, a = butter(2, [0.1, 0.7], btype='band', fs=500)
    bf = filtfilt(b, a, sig, axis=0)


    x = np.mean(bf, axis=1)
    nperseg = 128
    noverlap = int(nperseg * 0.95)
    nfft = nperseg * 16
    # Use Hamming window to match MATLAB default
    f_raw, t_seg, S = spectrogram(
        x, fs=500,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        mode='complex'
    )
    # Shift frequency axis
    S = np.fft.fftshift(S, axes=0)
    f = np.fft.fftshift(f_raw)
 
    WholeDuration = x.shape[0] / PRF
    t = np.linspace(0, WholeDuration, S.shape[1])

    # Convert to dB
    Sxx_db = 20 * np.log10(np.abs(S) / np.max(np.abs(S)))
    plt.pcolormesh(t, f, Sxx_db, shading='auto', cmap='jet')
    plt.clim(-40, 0)
    plt.ylim([-20, 20])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    print(f"Spectrogram saved to: {save_path}")

    # Load and preprocess the image for the model
    img = Image.open(save_path).convert('RGB')
    img = img.resize((64, 64))  # Adjust to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def main():
    MODEL_PATH = "/home/rpi/scripts_prisila/inference/distilled_student_model.pth"
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model.eval()

    shared_folder = "/shared_data"
    last_batch_id = None
    current_prediction = None
    first_run = True

    print("Starting continuous inference loop. Press Ctrl+C to cancel.")

    while True:
        batch_data, metadata = load_shared_data(shared_folder)
        if batch_data is None or metadata is None:
            if first_run:
                print("No data found.")
                first_run = False
            else:
                if current_prediction is not None:
                    print("No new data. Current prediction:", current_prediction)
                else:
                    print("No data found.")
        else:
            current_batch_id = metadata.get("batch_id")
            if current_batch_id != last_batch_id:
                print("\nNew batch detected:", metadata)

                # Start timing for inference latency
                start_time = time.time()

                # I_filtered, Q_filtered = process_batch(batch_data)
                img_tensor = process_batch(batch_data, fs=500.0, save_path="spectrogram.png")
                # img_tensor = preprocess_spectrogram(I_filtered, Q_filtered, fs=17.0, save_path="spectrogram.png")
                current_prediction = classify_spectrogram(img_tensor, model)

                # End timing
                latency_ms = (time.time() - start_time) * 1000

                # Gather system statistics
                cpu = psutil.cpu_percent(interval=0.1)
                ram = psutil.virtual_memory().percent
                temp_raw = os.popen("vcgencmd measure_temp").readline()
                temp_c = float(temp_raw.replace("temp=", "").replace("'C\n", ""))
                power_mw = ina.power if power_monitoring else "N/A"

                # Log all the metrics to the CSV log file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, current_batch_id, current_prediction,
                        f"{latency_ms:.2f}", cpu, ram, temp_c, power_mw
                    ])

                print(f"Prediction: {current_prediction} | Latency: {latency_ms:.2f} ms | CPU: {cpu}% | "
                      f"RAM: {ram}% | Temp: {temp_c}°C | Power: {power_mw} mW")

                last_batch_id = current_batch_id
            else:
                if current_prediction:
                    print("Prediction:", current_prediction)
                else:
                    print("No new batch detected.")

        # Loop sleep interval before checking again
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInference loop cancelled. Exiting.")
