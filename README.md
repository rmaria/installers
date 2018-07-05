# Keras + Tensorflow + Mxnet GPU install on ubuntu 16.04 

Much of this documentation is taken from: <br />
[https://github.com/williamFalcon/tensorflow-gpu-install-ubuntu-16.04]
   

These instructions are intended to set up a deep learning environment for GPU-powered tensorflow.      
[See here for pytorch GPU install instructions](https://github.com/williamFalcon/pytorch-gpu-install)

After following these instructions you'll have:

1. Ubuntu 16.04. 
2. Cuda 9.0 drivers installed.
3. A conda environment with python 3.6.    
4. The latest tensorflow version with gpu support.   

---   
### Step 0 minus 1: Install Ubuntu 16.04
```
I installed Ubuntu 16.04. The kernel is 4.13
$ uname -r
4.13.0-45-generic
```


---
### Step 0 minus 2: f you already have nvidia installed and want to remove it


```
sudo apt-get remove --purge nvidia-* 
sudo apt-get install ubuntu-desktop
sudo rm /etc/X11/xorg.conf
echo 'nouveau' | sudo tee -a /etc/modules`
```


### Step 0: Noveau drivers     
Before you begin, you may need to disable the opensource ubuntu NVIDIA driver called [nouveau](https://nouveau.freedesktop.org/wiki/).

**Option 1: Modify modprobe file**
1. After you boot the linux system and are sitting at a login prompt, press ctrl+alt+F1 to get to a terminal screen.  Login via this terminal screen.
2. Create a file: /etc/modprobe.d/nouveau
3.  Put the following in the above file...
```
blacklist nouveau
options nouveau modeset=0
```
4. reboot system   
```bash
reboot
```   
    
5. On reboot, verify that noveau drivers are not loaded   
```
lsmod | grep nouveau
```

If `nouveau` driver(s) are still loaded do not proceed with the installation guide and troubleshoot why it's still loaded.    

**Option 2: Modify Grub load command**    
From [this stackoverflow solution](https://askubuntu.com/questions/697389/blank-screen-ubuntu-15-04-update-with-nvidia-driver-nomodeset-does-not-work)    

1. When the GRUB boot menu appears : Highlight the Ubuntu menu entry and press the E key.
Add the nouveau.modeset=0 parameter to the end of the linux line ... Then press F10 to boot.   
2. When login page appears press [ctrl + ALt + F1]    
3. Enter username + password   
4. Uninstall every NVIDIA related software:   
```bash    
sudo apt-get purge nvidia*  
sudo reboot   
```   

---   
## Installation steps     


0. update apt-get   
``` bash 
sudo apt-get update
```
   
1. Install apt-get deps  
``` bash
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev curl   
```

2. install nvidia drivers 
``` bash
# The 16.04 installer works with 16.10.
# download drivers

This installer didn\'t work:
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb

(it was: cuda-repo-ubuntu1604_9.0.176-1_amd64.deb)

taken from: 
http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/
and it worked. The rest is the same


# download key to allow installation
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# install actual package
sudo dpkg -i ./cuda-repo-ubuntu1604_9.2.88-1_amd64.deb

#  install cuda (but it'll prompt to install other deps, so we try to install twice with a dep update in between
sudo apt-get update
sudo apt-get install cuda-9-2  
```    

2a. reboot Ubuntu
```bash
sudo reboot
```    

2b. check nvidia driver install 
``` bash
nvidia-smi   

# you should see a list of gpus printed    
# if not, the previous steps failed.   
``` 

For my case, it shows:
       

| NVIDIA-SMI 396.26                 Driver Version: 396.26                    |

| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |

|   0  GeForce GTX 107...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   57C    P0    35W /  N/A |    253MiB /  8119MiB |      0%      Default |

                                                                               

| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |

|    0      1082      G   /usr/lib/xorg/Xorg                           185MiB |
|    0      1925      G   compiz                                        64MiB |


<br />

3. Install cudnn   
``` bash
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```    

4. Add these lines to end of ~/.bashrc:   
``` bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```   

4a. Reload bashrc     
``` bash 
source ~/.bashrc
```   

5. Install miniconda   
``` bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   

# press s to skip terms   

# Do you approve the license terms? [yes|no]
# yes

# Miniconda3 will now be installed into this location:
# accept the location

# Do you wish the installer to prepend the Miniconda3 install location
# to PATH in your /home/ghost/.bashrc ? [yes|no]
# yes    

```   

5a. Reload bashrc     
``` bash 
source ~/.bashrc
```   

6. Create conda env to install keras, mxnet and tensorflow 
``` bash
conda create -n kemxtf

# environment location: /home/user/miniconda3/envs/kemxtf

```   

7. Activate env   
``` bash
source activate tensorflow   
```
```
Edit ~/.bashrc to add an alias:
alias activate='source activate'

source ~/.bashrc
```

8. Install tensorflow with GPU support for python 3.6    
``` bash
pip install tensorflow-gpu

# If the above fails, try the part below
# pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl
```   

9. Test tf install   
``` bash
# start python shell   
python

# run test script   
import tensorflow as tf   

hello = tf.constant('Hello, TensorFlow!')

# when you run sess, you should see a bunch of lines with the word gpu in them (if install worked)
# otherwise, not running on gpu
sess = tf.Session()
print(sess.run(hello))
```  

 
 10. Install mxnet   

In the virtual env
``` 
pip install mxnet
```

11. Test mxnet installation
in python:

>>> import mxnet as mx <br />
>>> a = mx.nd.ones((2, 3)) <br />
>>> b = a * 2 + 1  <br />
>>> b.asnumpy()  <br />

12. Install keras-mxnet

Do not use sudo, use just:

pip install keras-mxnet

13. Test installation

14. Install Opencv3

In the virtualenv 
```
conda install -c menpo opencv3 
```

15. Install Caffe

```
conda install -c anaconda caffe
```

