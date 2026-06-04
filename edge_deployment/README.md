# EdgeCaps Raspberry Pi Deployment

Here we present the Raspberry Pi deployment pipeline for EdgeCaps, an efficient Capsule Network designed for RF-based respiratory monitoring.

The deployment consists of:

1. RF radar data acquisition using the Novelda X4M200/x4M03 radar.
2. Real-time radar signal processing.
3. EdgeCaps model deployment on Raspberry Pi.
4. Logging and visualization of respiratory classification.


---

## Hardware Requirements

* Raspberry Pi 4
* Novelda X4M200 Radar
* MicroSD card (32 GB recommended)
* Raspberry Pi OS
* Network connectivity (Ethernet or Wi-Fi)

---

## Software Requirements

* Raspberry Pi OS
* Python 3.8+ (recommended for inference)
* Python 3.5 (required for radar operation)


---


### Installation

### Clone Repository

```bash
git clone https://github.com/<prisila20/<repo_name>.git
cd <repo_name>
```

### Create Virtual Environment

```bash
python3 -m venv inference-env
source inference-env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Deployment

Place trained model weights inside:

```text
models/
```

---

## Running Real-Time Inference

Activate the environment:

```bash
source inference-env/bin/activate
```

Navigate to the inference directory:

```bash
cd inference
```

Run:

```bash
python log_final_inference.py
```

or

```bash
python realtime_inference.py
```

---

## Radar Data Acquisition

Radar data collection is performed using the Novelda X4M03 ModuleConnector interface.

Run:

```bash
python complete_radar_writer.py
```

Collected radar frames are written to the shared data directory and accessed by the inference pipeline.

---

## Shared Data Architecture

The deployment uses a shared directory for communication between:

* Radar raw data acquisition process
* Respiration anomaly classification

```text
Radar Writer >> shared_data >> inference pipeline ```


---

## Performance Monitoring

The deployment logs:

* Predictions
* Timestamps
* Processing latency
* System status

Logs can be used for deployment benchmarking and validation.

---

## Documentation

Detailed setup instructions are available in:

```text
docs/raspberry_pi_setup.md
```

including:

* Python 3.5 build process
* ARMHF chroot environment
* Boost installation
* Novelda ModuleConnector installation
* Network configuration
* Shared directory setup

---

## Citation





