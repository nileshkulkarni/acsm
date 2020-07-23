## Imagenet Quadurped Dataset
We collected a dataset of foreground mask annotations for quadrupeds from imagenet. We provide the annotations collected from workers on the internet. 

We only provide annotations for non-occulded and non-truncated object quadrupeds from the following classes on Imagenet along with the synset ids. We also provide instance segmentation outputs from our quadruped segmentation network. 


| Category        | ImageNet SynsetIDs                         |
| --------------- | ------------------------------------------ |
| rhino           | n02391994                                  |
| giraffe         | n02439033                                  |
| camel           | n02437312                                  |
| hippo           | n02398521                                  |
| fox             | n02119022, n02119789, n02120079, n02120505 |
| bear            | n02132136, n02133161, n02131653            |
| leopard         | n02128385                                  |
| bison           | n02410509                                  |
| buffalo         | n02408429, n02410702                       |
| donkey          | n02390640, n02390738                       |
| goat            | n02416519, n02417070                       |
| beest           | n02421449, n02422106                       |
| kangaroo        | n01877812                                  |
| german-shepherd | n02106662, n02107574, n02109047            |
| pig             | n02396427, n02395406, n02397096            |
| lion            | n02129165                                  |
| llama           | n02437616,  n02437971                      |
| tapir           | n02393580,  n02393940                      |
| tiger           | n02129604                                  |
| warthog         | n02397096                                  |
| wolf            | n02114367, n02114548, n02114712            |


---
## Download the dataset

Filtered images that have non-occluded and non-truncated instances  of quadrupeds: [Here](http://fouheylab.eecs.umich.edu/~nileshk/acsm_data/imquad_release/quads74K_pos.csv)

Annoatations in coco-format :  [Here](http://fouheylab.eecs.umich.edu/~nileshk/acsm_data/imquad_release/annotations/im_quads_coco.json). Quadrupeds have the "*person*" as the class label.

Inferred Annotations for all images in the above categories: [Here]( ) (~ approximately 20G of mask predictions using [point-rend](http://fouheylab.eecs.umich.edu/~nileshk/acsm_data/imquad_release/masks_point_rend.tar))

---
## Using the Pre-trained network weights

Download and setup detectron2 and point-rend as per the setup instructions [Here](https://github.com/facebookresearch/detectron2). Apply the patch file changes from `edit_content.diff`. For the ease and similicity of usage we have relabeled "quadruped" class as "person" as it allows to use the pre-defined code and coco annotations as it is.  



**Setup Instructions**
```
DETECTRON_DIR = /detectron2/path/
cd $DETECTRON_DIR
git checkout bd2ea475b693a88c063e05865d13954d50242857
patch -p0 -i edit_content.diff  ## Applies the patch changes to detectron2
POINT_REND_DIR = $DETECTRON_DIR/projects/PointRend/
mkdir -p $POINT_REND_DIR/datasets/im_quad/annotations
cd $POINT_REND_DIR/datasets/im_quad/annotations
wget http://fouheylab.eecs.umich.edu/~nileshk/acsm_data/imquad_release/annotations/im_quads_coco.json  im_quads_coco.json
```

**Download and run pretrained model**

```
cd $POINT_REND_DIR
mkdir output
wget http://fouheylab.eecs.umich.edu/~nileshk/acsm_data/imquad_release/point_rend/model_0029999.pth model_0029999.pth
sh instance_segment.sh <input_img_dir> <output_mask_dir>
```






