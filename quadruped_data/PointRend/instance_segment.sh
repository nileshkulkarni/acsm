imdir=$1
outdir=$2
echo 'Loading image from ' $imdir
echo 'Saving masks to ' $outdir
mkdir -p $outdir

python instance_segment.py --config-file configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_imquad.yaml  --input $imdir --output $outdir --opts MODEL.WEIGHTS output/model_0029999.pth


