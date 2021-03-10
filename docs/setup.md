## Setup annotations and data directories


### Download template models and annotations 
Download from [here](https://www.dropbox.com/s/3tj037gnk4gz11t/cachedir.tar?dl=0)
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
Download our pretrained model and cached annotations from [here](https://www.dropbox.com/s/8zo5ohwhv99efqr/cachedir.tar.gz?dl=0)
```
cd acsm
tar -xf cachedir.tar
```
### Download models for other imagenet catgories
```
cd acsm/cachedir/
wget https://www.dropbox.com/s/05lohn7x96o3fuf/models.zip?dl=0
unzip -q models.zip 
```
#### Training and Testing on CUBS dataset
* Train Birds with Keypoints. Generate training command using this 
  ```
    python -m acsm.experiments.job_script --category=bird --kp=True --parts_file=acsm/part_files/bird.txt
  ```

* Train Birds without Keypoints
  ```
    python -m acsm.experiments.job_script --category=bird --kp=False --parts_file=acsm/part_files/bird.txt
  ```


* Evaluate KP Projection
    ```
    python -m acsm.benchmark.pascal.kp_project --name=acsm_bird_3parts --category=bird --parts_file=acsm/part_files/bird.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_bird_3parts_pck_val --multiple_cam=True  --visuals_freq=5 --visualize=True --n_data_workers=4 --scale_bias=1.5  --resnet_style_decoder=True --resnet_blocks=4 --el_euler_range=90 --cyc_euler_range=60
    ```


* Evaluate KP PCK Transfer
    ```
    python -m acsm.benchmark.pascal.kp_transfer --name=acsm_bird_3parts --category=bird  --parts_file=acsm/part_files/bird.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_bird_3parts_transfer_pck_val --multiple_cam=True --num_eval_iter=10000 --visuals_freq=1000  --visualize=True --n_data_workers=4  --scale_bias=1.5  --resnet_style_decoder=True  --resnet_blocks=4 --el_euler_range=90 --cyc_euler_range=60
    ```



#### Training and Testing on Pascal Horses dataset
* Train Horses with Keypoints. Generate training command using this 
  ```
    python -m acsm.experiments.job_script --category=horse --kp=True --parts_file=acsm/part_files/horse.txt
  ```

* Train Horses without Keypoints
  ```
    python -m acsm.experiments.job_script --category=horse --kp=False --parts_file=acsm/part_files/horse.txt
  ```


* Evaluate KP PCK
    ```
    python -m acsm.benchmark.pascal.kp_project --name=acsm_horse_8parts --category=horse --parts_file=acsm/part_files/horse.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_horse_8parts_pck_val --multiple_cam=True  --visuals_freq=5 --visualize=True --n_data_workers=4 --scale_bias=0.75 --resnet_style_decoder=True --resnet_blocks=4 --el_euler_range=20 --cyc_euler_range=20
    ```


* Evaluate KP Projection
    ```
    python -m acsm.benchmark.pascal.kp_transfer --name=acsm_horse_8parts --category=horse  --parts_file=acsm/part_files/horse.txt --use_html --dl_out_pascal=True --dl_out_imnet=False --split=val --num_train_epoch=200 --num_hypo_cams=8 --env_name=acsm_horse_8parts_transfer_pck_val --multiple_cam=True --num_eval_iter=10000 --visuals_freq=1000  --visualize=True --n_data_workers=4  --scale_bias=0.75  --resnet_style_decoder=True  --resnet_blocks=4 --el_euler_range=20 --cyc_euler_range=20
    ```


#### Other configurations of models.

| Model                | Keypoint Supv | Num of Parts |
| -------------------- | ------------- | ------------ |
| acsm_bird_kp_3parts  | Yes           | 3            |
| acsm_bird_3parts     | No            | 3            |
| acsm_bird_kp_0parts  | Yes           | 0            |
| acsm_bird_0parts     | No            | 0            |
|                      |               |              |
| acsm_horse_kp_8parts | Yes           | 8            |
| acsm_horse_8parts    | No            | 8            |
| acsm_horse_kp_0parts | Yes           | 0            |
| acsm_horse_0parts    | No            | 0            |


