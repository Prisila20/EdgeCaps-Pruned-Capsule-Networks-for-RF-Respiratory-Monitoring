# Raspberry Pi Deployment Setup

## Overview

The Novelda X4M03/X4M200 radar relies on the Legacy-SW ModuleConnector library, for which the available precompiled binaries are compatible only with Python 3.5. However, the deep learning inference pipeline uses PyTorch, which requires a newer Python version (Python 3.8+ on Raspberry Pi).

To enable both systems to operate on the same Raspberry Pi, this project uses a dual-environment architecture:

* **Python 3.5 ARMHF environment** for radar communication through ModuleConnector.
* **Python 3.8+ environment** for real-time model inference using PyTorch.
* A **shared data directory** for communication between the radar acquisition process and the inference process.

An ARMHF chroot environment is used to isolate the legacy Python 3.5 dependencies while allowing the host Raspberry Pi operating system to run a modern Python environment for machine learning inference.

The deployment architecture is illustrated below:

```text
Novelda X4M200 Radar
          │
          ▼
 Python 3.5 ARMHF Chroot
 (ModuleConnector)
          │
          ▼
     Shared Data
          │
          ▼
 Python 3.8+ Environment
   (PyTorch Inference)
          │
          ▼
 Respiratory Prediction
```

This document provides a step-by-step guide for reproducing the deployment environment.

---

## 1. Network Configuration

Configure network connectivity and verify internet access.

---

## 2. Python 3.5 Build

Build and install Python 3.5.10 required by the Novelda ModuleConnector library.

**Update System and Install Build Dependencies** 
```
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget \
  curl \
  llvm \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libffi-dev \
  liblzma-dev \
  libgdbm-dev \
  libnss3-dev \
  libgdbm-compat-dev
```

***Download Python 3.5.10***
```
cd /usr/src

sudo wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz

sudo tar xzf Python-3.5.10.tgz

cd Python-3.5.10
```

***Build and Install***
```
sudo ./configure --enable-optimizations --enable-shared

sudo make -j$(nproc)

sudo make altinstall

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/python3.5.conf

sudo ldconfig
```
***Create Virtual Environment and install pip***
```
python3.5 -m venv ~/my35env

source ~/my35env/bin/activate

curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py

python get-pip.py
```

---

## 3. ARMHF Chroot Creation

Create a 32-bit ARMHF environment to support the legacy radar software stack.

---

## 4. Boost Installation

Install Boost 1.62.0 and associated dependencies required by ModuleConnector.

---

## 5. Radar ModuleConnector Installation

Install and verify the Novelda Legacy-SW ModuleConnector package.

---

## 6. Shared Data Configuration

Configure bind-mounted shared directories used for communication between radar acquisition and inference processes.

---

## 7. Inference Environment Setup

Create a Python 3.8+ virtual environment and install PyTorch and inference dependencies.

---

## 8. System Operation

### Start Radar Acquisition

Launch the radar writer process inside the Python 3.5 ARMHF environment.

### Start Inference

Launch the inference pipeline in the Python 3.8 environment.

### Verify Operation

Confirm that radar frames are being written to the shared directory and consumed by the inference process.

---

## 9. Troubleshooting

Common issues and solutions for:

* Network connectivity
* Chroot configuration
* ModuleConnector installation
* Shared directory mounting
* PyTorch installation
* Real-time inference
