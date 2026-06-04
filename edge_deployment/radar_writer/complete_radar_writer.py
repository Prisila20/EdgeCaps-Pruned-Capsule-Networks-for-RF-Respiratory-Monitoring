#!/usr/bin/env python3
import os
import sys
import time
import json
import glob
import numpy as np
from time import sleep
import pymoduleconnector
from pymoduleconnector import DataType

# Configure shared data directory
SHARED_DIR = "/shared_data"
os.makedirs(SHARED_DIR, exist_ok=True)

# Radar parameters
FPS = 17
ITERATIONS = 16
PULSES_PER_STEP = 300
DAC_MIN = 949
DAC_MAX = 1100
AREA_START = 0.4
AREA_END = 5

# Batch collection parameters
BATCH_DURATION = 15  # seconds
FRAMES_PER_BATCH = int(FPS * BATCH_DURATION)  # ~255 frames at 17 FPS

def find_radar_device():
    """
    Try to find the radar device on Linux (Raspberry Pi).
    Returns the device path if found, else None.
    """
    # Check for a symlink first
    if os.path.exists("/dev/radar_device"):
        print("Found radar device at /dev/radar_device")
        return "/dev/radar_device"
    
    # Look for ttyACM devices
    devices = glob.glob('/dev/ttyACM*')
    if devices:
        print("Found USB serial device:", devices[0])
        return devices[0]
    
    # Look for ttyUSB devices as a fallback
    devices = glob.glob('/dev/ttyUSB*')
    if devices:
        print("Found USB device:", devices[0])
        return devices[0]
    
    print("No radar device found!")
    return None

class XepData:
    def __init__(self, device_name, fps, iterations, pulses_per_step, dac_min, dac_max, area_start, area_end):
        self.device_name = device_name
        self.FPS = fps
        self.iterations = iterations
        self.pulses_per_step = pulses_per_step
        self.dac_min = dac_min
        self.dac_max = dac_max
        self.area_start = area_start
        self.area_end = area_end
        
        # Calculate additional parameters (if needed for further processing)
        self.bin_length = 8 * 1.5e8 / 23.328e9
        self.fast_sample_point = int((self.area_end - self.area_start) / self.bin_length + 2)
        
        # For metadata
        self.sample_rate = fps
        self.range_start = area_start
        self.range_end = area_end
        
        print("Connecting to device {}...".format(self.device_name))
        self.mc = pymoduleconnector.ModuleConnector(self.device_name)
        self.xep = self.mc.get_xep()
        print("Connected. Initializing system...")
        
        self.sys_init()
        print("System initialized successfully")
    
    def sys_init(self):
        """Initialize the radar system with the configured parameters."""
        self.xep.x4driver_init()
        sleep(0.1)
        self.xep.x4driver_set_downconversion(1)
        sleep(0.1)
        self.xep.x4driver_set_iterations(self.iterations)
        sleep(0.1)
        self.xep.x4driver_set_pulses_per_step(self.pulses_per_step)
        sleep(0.1)
        self.xep.x4driver_set_dac_min(self.dac_min)
        sleep(0.1)
        self.xep.x4driver_set_dac_max(self.dac_max)
        sleep(0.1)
        self.xep.x4driver_set_frame_area_offset(0.18)
        sleep(0.1)
        self.xep.x4driver_set_frame_area(self.area_start, self.area_end)
        sleep(0.1)
        self.xep.x4driver_set_fps(self.FPS)
        sleep(0.1)
        
        # Clear any residual messages in the buffer
        self.clear_buffer()
    
    def clear_buffer(self):
        """Clear any pending data in the buffer."""
        count = 0
        while self.xep.peek_message_data_float():
            self.xep.read_message_data_float()
            count += 1
        if count > 0:
            print("Cleared {} messages from buffer".format(count))
    
    def read_apdata(self):
        """Read and return the I/Q data from the radar."""
        try:
            data_msg = self.xep.read_message_data_float()
            data = data_msg.data
            data_length = len(data)
            # Split the data into I and Q parts
            half = data_length // 2
            i_vec = np.array(data[:half])
            q_vec = np.array(data[half:])
            return i_vec, q_vec
        except Exception as e:
            print("Error reading data: {}".format(e))
            return np.array([]), np.array([])
    
    def display_sys_info(self):
        """Display system information from the radar module."""
        print("FirmWareID =", self.xep.get_system_info(2))
        print("Version =", self.xep.get_system_info(3))
        print("Build =", self.xep.get_system_info(4))
        print("VersionList =", self.xep.get_system_info(7))
    
    def close(self):
        """Properly close the connection to the radar module."""
        try:
            self.mc.close()
            print("Radar connection closed")
        except Exception as e:
            print("Error closing connection: {}".format(e))
    
    def __del__(self):
        try:
            self.close()
        except:
            pass

def main():
    print("XeThru Radar Batch Data Collection")
    print("----------------------------------")
    print("Collecting batches of approximately {} seconds ({} frames)".format(BATCH_DURATION, FRAMES_PER_BATCH))
    
    # Find radar device
    device_path = find_radar_device()
    if not device_path:
        print("No radar device found. Please check connections and try again.")
        return 1
    
    print("Found radar device at {}".format(device_path))
    
    # Initialize the radar system
    try:
        radar = XepData(device_path, FPS, ITERATIONS, PULSES_PER_STEP, DAC_MIN, DAC_MAX, AREA_START, AREA_END)
        radar.display_sys_info()
        
        # Allow extra time for stabilization
        print("Waiting for radar to stabilize...")
        sleep(3)
        
        print("\nStarting batch data collection and writing to {}...".format(SHARED_DIR))
        print("Press Ctrl+C to stop\n")
        
        batch_count = 0
        while True:
            batch_count += 1
            batch_start_time = time.time()
            print("Starting batch #{}...".format(batch_count))
            
            i_data_batch = []
            q_data_batch = []
            frame_count = 0
            
            # Collect frames for this batch
            while frame_count < FRAMES_PER_BATCH:
                i_vec, q_vec = radar.read_apdata()
                # Skip empty frames
                if i_vec.size == 0 or q_vec.size == 0:
                    print("Empty frame received, skipping...")
                    sleep(0.1)
                    continue
                i_data_batch.append(i_vec)
                q_data_batch.append(q_vec)
                frame_count += 1
                
                if frame_count % 50 == 0:
                    print("  Collected {}/{} frames...".format(frame_count, FRAMES_PER_BATCH))
                
                # Control frame rate
                elapsed = time.time() - (batch_start_time + (frame_count - 1) / FPS)
                sleep_time = max(0, (1 / FPS) - elapsed)
                sleep(sleep_time)
            
            # Convert collected frames to numpy arrays
            batch_i_data = np.array(i_data_batch)
            batch_q_data = np.array(q_data_batch)
            timestamp = int(time.time() * 1000)
            
            # Define file paths
            current_file = os.path.join(SHARED_DIR, "current.npy")
            previous_file = os.path.join(SHARED_DIR, "previous.npy")
            
            # Rotate files: move current to previous if it exists
            if os.path.exists(current_file):
                if os.path.exists(previous_file):
                    os.remove(previous_file)
                os.rename(current_file, previous_file)
            
            # Save new batch data
            batch_data = {"i_data": batch_i_data, "q_data": batch_q_data}
            np.save(current_file, batch_data)
            
            # Save metadata about the batch
            metadata = {
                "timestamp": timestamp,
                "batch_id": batch_count,
                "frames": frame_count,
                "duration": time.time() - batch_start_time,
                "parameters": {
                    "fps": radar.FPS,
                    "range_start": radar.area_start,
                    "range_end": radar.area_end,
                    "i_shape": batch_i_data.shape,
                    "q_shape": batch_q_data.shape
                },
                "current_file": "current.npy",
                "previous_file": "previous.npy" if os.path.exists(previous_file) else None
            }
            with open(os.path.join(SHARED_DIR, "batch_info.json"), 'w') as f:
                json.dump(metadata, f)
            
            batch_time = time.time() - batch_start_time
            print("Batch #{} complete: {} frames in {:.2f}s".format(batch_count, frame_count, batch_time))
            print("Average FPS: {:.2f}".format(frame_count / batch_time))
            print("Data saved to {}".format(current_file))
            print("Metadata saved to {}/batch_info.json".format(SHARED_DIR))
            print("----------------------------------")
            
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
        print("Collected {} batches".format(batch_count))
    except Exception as e:
        print("Error during data collection: {}".format(e))
        return 1
    finally:
        try:
            radar.close()
        except Exception as e:
            print("Error closing radar: {}".format(e))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
