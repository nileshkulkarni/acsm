# Articulation-Aware Canonical Surface Mapping
Nilesh Kulkarni, Abhinav Gupta, David F. Fouhey, Shubham Tulsiani

[Paper](https://arxiv.org/pdf/2004.00614.pdf)
[Project Page](https://nileshkulkarni.github.io/acsm/)

<img src="https://nileshkulkarni.github.io/acsm/resources/images/teaser.png" width="60%">

## Requirements
* Python 2.7
* PyTorch tested with `1.2.0` and works with `1.3.0` too.
* cuda 9.2

For setup and installation refer to [docs/install.md](docs/install.md) instructions.


## Setup Evlaution and Training
For ease of acess we provide python scripts that can generate slurm scripts that can be used to generate the results in the paper.

* Downloading pre-trained model and annotations. Follow setup instructions [here](docs/setup.md)

* Training from scratch.  Follow setup instructions [here](docs/setup.md)


## Citation
If you find the code useful for your research, please consider citing:-
```
@inproceedings{kulkarni2020articulation,
  title={Articulation-aware Canonical Surface Mapping},
  author={Kulkarni, Nilesh and Gupta, Abhinav and Fouhey, David F and Tulsiani, Shubham},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={452--461},
  year={2020}
}
```



## Future Release
* Python 3.6
* PyTorch 1.5
* PyTorch3D

