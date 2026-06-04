# Raspberry Pi Deployment Setup

## Overview

The Novelda X4M03/X4M200 radar relies on the Legacy-SW ModuleConnector library, for which the available precompiled binaries are compatible only with Python 3.5. However, the deep learning inference pipeline uses PyTorch, which requires a newer Python version (Python 3.8+ on Raspberry Pi).

To enable both systems to operate on the same Raspberry Pi, this project uses a dual-environment architecture:

* **Python 3.5 ARMHF environment** for radar communication through ModuleConnector.
* **Python 3.8+ environment** for real-time model inference using PyTorch.
* A **shared data directory** for communication between the radar acquisition process and the inference process.

An ARMHF chroot environment is used to isolate legacy Python 3.5 dependencies while allowing the host Raspberry Pi operating system to run a modern Python environment for machine-learning inference.

The deployment architecture is illustrated below:

```text
Novelda X4M200 Radar
          │
          ▼
 Python 3.5 ARMHF Chroot (32-bit)
 (ModuleConnector)
          │
          ▼
     Shared Data
          │
          ▼
 Python 3.8+ Environment Host Pi OS (64-bit)
   (PyTorch Inference)
          │
          ▼
 Respiratory Classification
```

This document provides a step-by-step guide for reproducing the deployment environment.

---

## 1. Network Configuration

Configure network connectivity and verify internet access.

---

## 2. Software installation on Host Pi

Install Python 3.8 or a newer version on the Raspberry Pi host operating system. Create a virtual environment and install the dependencies required by the packages listed in the project's requirements.txt file:

```
pip install --upgrade pip

pip install -r requirements.txt
```
This environment will later be used to run the real-time respiratory inference pipeline, while raw radar data acquisition is handled separately within the ARMHF chroot environment.

---

## 3. ARMHF Chroot Creation

Create a 32-bit ARMHF environment to support the legacy radar software stack. The ModuleConnector binaries are built for 32-bit ARM (ARMHF). To support them on a modern Raspberry Pi system, an ARMHF chroot environment is created.

***Install Chroot Tools and Helpers***
```
sudo apt-get update
sudo apt-get install -y debootstrap qemu-user-static binfmt-support
sudo mkdir -p /srv/pi32-chroot #Create a new 32-bit Chroot Directory
```
***Bootstrap ARMHF Debian Environment***
```
sudo debootstrap --arch=armhf bookworm /srv/pi32-chroot http://deb.debian.org/debian
```
***Copy QEMU Emulator***
```
sudo cp /usr/bin/qemu-arm-static /srv/pi32-chroot/usr/bin/
```
***Mount Required Filesystems*** 
```
sudo mount -o bind /dev  /srv/pi32-chroot/dev
sudo mount -o bind /proc /srv/pi32-chroot/proc
sudo mount -o bind /sys  /srv/pi32-chroot/sys
sudo cp /usr/bin/qemu-arm-static /srv/pi32-chroot/usr/bin/
```
***Enter Chroot and install packages***
```
sudo chroot /srv/pi32-chroot /bin/bash
#Install development packages
apt-get update
apt-get install -y libffi-dev:armhf
apt-get install -y build-essential libssl-dev zlib1g-dev \
                   libsqlite3-dev libbz2-dev libreadline-dev libncursesw5-dev \
                   wget
```
***Python 3.5 Installation and Settings***
```
cd /tmp
wget https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz
tar xf Python-3.5.10.tgz
cd Python-3.5.10

#./configure --prefix=/usr/local/python3.5 --enable-shared
CC=arm-linux-gnueabihf-gcc \
./configure --host=arm-linux-gnueabihf --build=arm-linux-gnueabihf \
            --prefix=/usr/local/python3.5 --enable-shared

make -j4
make altinstall

#set python 3.5 path
export LD_LIBRARY_PATH=/usr/local/python3.5/lib:$LD_LIBRARY_PATH

#set dynamic linker permanently
echo "/usr/local/python3.5/lib" > /etc/ld.so.conf.d/python3.5.conf
ldconfig
/usr/local/python3.5/bin/python3.5
/usr/local/python3.5/lib/libpython3.5m.so.1.0

/usr/local/python3.5/bin/python3.5 -c "import _ctypes; print(_ctypes)"
```
---

## 4. Boost Installation

Install Boost 1.62.0 and the associated dependencies required by ModuleConnector.
```
cd ~tmp/
wget "https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.gz/download" -O boost_1_62_0.tar.gz
tar xzf boost_1_62_0.tar.gz
cd /tmp/boost_1_62_0

./bootstrap.sh
./b2 -j$(nproc) install --with-python --with-filesystem \
   include=/usr/local/python3.5/include/python3.5m \
   library-path=/usr/local/python3.5/lib \
   cxxflags="-fPIC -std=c++11 -Wno-deprecated-declarations"

#update dynamic linker
ldconfig
#verify
ls /usr/local/lib | grep libboost_filesystem
```
---

## 5. Radar ModuleConnector Installation
On the Raspberry Pi host system, clone or download the Legacy-SW repository from https://github.com/novelda/Legacy-SW/tree/master/ModuleConnector. Then follow the steps below to install it in the chroot environment.

```
#From host system:
sudo mkdir -p /srv/pi32-chroot/mnt/pymoduleconnector
sudo mount --bind \
  /home/rpi/Legacy-SW/ModuleConnector/ModuleConnector-rpi-1/python35-arm-linux-gnueabihf \
  /srv/pi32-chroot/mnt/pymoduleconnector
```

Enter chroot to install moduleconnectors
```
sudo linux32 chroot /srv/pi32-chroot /bin/bash
cd /mnt/pymoduleconnector
/usr/local/python3.5/bin/python3.5 setup.py install

#Installing numpy
/usr/local/python3.5/bin/python3.5 -m pip install --upgrade pip==20.3.4 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

/usr/local/python3.5/bin/python3.5 -m pip install "Cython<0.30" --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

/usr/local/python3.5/bin/python3.5 -m pip install "numpy<1.20" --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

```
Mount all support modules to chroot from the host system
```
sudo mount --bind /dev   /srv/pi32-chroot/dev
sudo mount --bind /dev/pts /srv/pi32-chroot/dev/pts
sudo mount --bind /proc  /srv/pi32-chroot/proc
sudo mount --bind /sys   /srv/pi32-chroot/sys
# Create a mountpoint inside the chroot (on the host filesystem)
sudo mkdir -p /srv/pi32-chroot/mnt/radar_scripts

#Bind-mount your host's radar_working folder into the chroot
sudo mount --bind \
    /home/rpi/radar_writer \
    /srv/pi32-chroot/mnt/radar_scripts     #change folder names accordingly
```

---

## 6. Shared Data Configuration

The radar acquisition process and inference process run in separate environments. A shared directory is used for communication.
```
#From the host Pi environment
sudo mkdir -p /srv/pi32-chroot/shared_data
sudo mkdir -p /shared_data
```

Bind mount the shared directory

```
sudo mount --bind /srv/pi32-chroot/shared_data /shared_data
```

Configure Persistent Mount:

```
sudo nano /etc/fstab
```
Add:

```
/srv/pi32-chroot/shared_data   /shared_data   none   bind   0   0 
```

Apply changes:
```
sudo mount -a
```
---

## 7. System Operation

***Start Radar Acquisition***
Enter the ARMHF chroot:
```
sudo chroot /srv/pi32-chroot /bin/bash
#navigate to radar control script
cd /mnt/radar_scripts
```
Start data acquisition:
```
/ usr/local/python3.5/bin/python3.5 complete_radar_writer.py
```
The script continuously writes radar frames to the shared directory.

### Model Inference
Open a second terminal. Activate the inference environment and navigate to the inferencing script, then run the model inference script:
```
python log_final_inference.py
```
The inference pipeline monitors the shared directory, processes incoming radar frames, and generates a respiratory state class in real time.

---
