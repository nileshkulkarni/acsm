# Installation Instruction
Needs a cuda9.2 for installation. CuPy and Chainer will not work otherwise.

## Download code
```
git clone https://github.com/nileshkulkarni/acsm.git
```

## Setup Conda Env
* Create Conda environment
```
conda create -n acsm python=2.7
conda activate acsm
pip install numpy==1.16.6
```


* Install Cupy, make sure CUDA_PATH and CUDA_HOME are pointing to cuda9.2 location
```
mkdir acsm/acsm/external/sources -p
cd acsm/acsm/external/sources
pip download cupy==2.3.0
tar -xf cupy-2.3.0.tar.gz
cd cupy-2.3.0
python setup.py install 
```

* Install other packages and dependencies
Refer [here](https://pytorch.org/get-started/previous-versions/) for pytorch installation
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
conda install -c anaconda scikit-image imageio -y
pip install -r acsm/requirements.txt
```

* Setup Neural Mesh Renderer
```
cd acsm/acsm/external/
git clone https://github.com/hiroharu-kato/neural_renderer --branch v1.1.0
cd neural_renderer
python setup.py install
```