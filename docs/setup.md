## Setup annotations and data directories


### Download template models and annotations
We provide rigged version of 3D template models for articulation. Every model has the following structured
```
cachedir
└───models
│   └───horse
│       │   mean_shape.mat ## contains mapping from UV value to faceindex on the template shape 
│       │   model.obj ## Base template shape
│       │   kp2vertex.txt ## Approximate 3D locations for keypoint vertices
│       │   mirror_transforms.txt ## Correspondence between transformation on reflection
│       │   hierarchy.xml ## Part hierarcy for articulation
│       │   parts.pkl ## Labelled vertices for every part
│       │   part_names.txt ## Part names
│   
└───bird
    │    ...

```


### Download image annotations and splits


#### Training and Testing on CUBS dataset
Download our pretrained model and cached annotations
```
cd acsm
wget <anno_path_cachedir> 
cd 
tar -xf cachedir.tar
```


* Train Birds with Keypoints. Generate training command using this 
  ```
    python -m acsm.experiments.job_script --category=bird --kp=True --parts_file=dcsm/part_files/bird.txt
  ```

* Train Birds without Keypoints
  ```
    python -m acsm.experiments.job_script --category=bird --kp=False --parts_file=dcsm/part_files/bird.txt
  ```


* Evaluate KP PCK
    ```
    python -m acsm.benchmark.pascal.kp_project --name=acsm_bird_3parts --category=bird --parts_file=dcsm/part_files/bird.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_bird_3parts_pck_val --multiple_cam=True  --visuals_freq=5 --visualize=True --n_data_workers=4 --scale_bias=1.5  --resnet_style_decoder=True --resnet_blocks=4
    ```


* Evaluate KP Projection
    ```
    python -m acsm.benchmark.pascal.kp_transfer --name=acsm_bird_3parts --category=bird  --parts_file=dcsm/part_files/bird.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_bird_3parts_transfer_pck_val --multiple_cam=True --num_eval_iter=10000 --visuals_freq=1000  --visualize=True --n_data_workers=4  --scale_bias=1.5  --resnet_style_decoder=True  --resnet_blocks=4
    ```
