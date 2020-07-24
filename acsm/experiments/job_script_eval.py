from absl import app, flags
import pdb
flags.DEFINE_string('suffix', '', 'suffix to name')
flags.DEFINE_boolean('kp', False, 'without kp')
flags.DEFINE_string('parts_file', 'dcsm/part_files/bird.txt', 'parts file')
flags.DEFINE_string('category', 'bird', 'bird')


def birds_kp(
    name,
    parts_file,
):
    command = "python -m dcsm.experiments.pascal.csp --name {} --batch_size=12 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=3.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=200 --warmup_pose_iter=500 --warmup_deform_iter=3000 --warmup_semi_supv=0 --multiple_cam=True --flip_train=False --ent_loss_wt=0.05 --scale_bias=1.5 --num_hypo_cams=1 --parts_file={} --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=False --resnet_blocks=4 --depth_loss_wt=1.0 --category=bird".format(
        name, parts_file
    )
    return command


def birds_no_kp(
    name,
    parts_file,
):
    command = "python -m dcsm.experiments.pascal.csp --name {} --batch_size=6 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=0.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=200 --warmup_pose_iter=2000 --warmup_deform_iter=7000 --warmup_semi_supv=0 --multiple_cam=True --flip_train=True --ent_loss_wt=0.05 --scale_bias=1.5 --num_hypo_cams=8 --parts_file={} --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=True --category=bird --resnet_blocks=4 --depth_loss_wt=1.0  --category=bird --el_euler_range=90 --cyc_euler_range=60".format(
        name, parts_file
    )
    return command


def horse_kp(
    name,
    parts_file,
):
    command = "python -m dcsm.experiments.pascal.csp --name {} --batch_size=12 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=3.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=800 --warmup_pose_iter=500 --warmup_deform_iter=3000  --reproject_loss_wt=10.0 --warmup_semi_supv=1000 --multiple_cam=True --flip_train=False --ent_loss_wt=0.05 --scale_bias=0.75 --num_hypo_cams=1 --parts_file={} --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=False --depth_loss_wt=1.0 --resnet_blocks=4 --category=horse".format(
        name, parts_file
    )
    return command


def horse_no_kp(
    name,
    parts_file,
):
    command = "python -m dcsm.experiments.pascal.csp --name {} --batch_size=6 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=0.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=400 --warmup_pose_iter=500 --warmup_deform_iter=10000 --warmup_semi_supv=0 --multiple_cam=True --flip_train=True --ent_loss_wt=0.05 --scale_bias=0.75 --num_hypo_cams=8 --parts_file={} --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=True --resnet_blocks=4 --depth_loss_wt=1.0 --category=horse --el_euler_range=20 --cyc_euler_range=20".format(
        name, parts_file
    )
    return command


def category_no_kp(
    category,
    name,
    parts_file,
):
    command = "python -m dcsm.experiments.pascal.csp --name {} --batch_size=6 --n_data_workers=8 --display_port=8098 --display_visuals --display_freq=100 --save_visuals --save_visual_freq=100 --use_html --kp_loss_wt=0.0 --save_epoch_freq=50 --save_visual_count=1 --single_axis_pred=True --dl_out_pascal=True --dl_out_imnet=True --num_epochs=400 --warmup_pose_iter=500 --warmup_deform_iter=10000 --warmup_semi_supv=0 --multiple_cam=True --flip_train=True --ent_loss_wt=0.05 --scale_bias=0.75 --num_hypo_cams=8 --parts_file={} --reproject_loss_wt=1.0 --mask_loss_wt=0.0 --no_trans=False --cov_mask_loss_wt=10 --con_mask_loss_wt=0.1 --n_contour=1000  --nmr_uv_loss_wt=0.0 --resnet_style_decoder=True --resnet_blocks=4 --depth_loss_wt=1.0 --category={} --el_euler_range=20 --cyc_euler_range=20".format(
        name, parts_file, category
    )
    return command


FLAGS = flags.FLAGS


def get_sbatch_header(name):
    header = """#!/bin/sh
#SBATCH --partition=vl-fb-gtx1080
#SBATCH --job-name={}
#SBATCH --output=/home/nileshk/DeformParts/refactor/dcsm/cachedir/slurm_logs/{}.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00
#SBATCH --mem=16G
set -x 

module load cuda/9.2
cd /home/nileshk/DeformParts/refactor/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dcsm2
hostname""".format(name, name)
    return header


def main(_):
    category = FLAGS.category
    name = FLAGS.suffix
    kp = FLAGS.kp
    if name == '':
        name = "acsm_py2pt7_{}_kp_{}".format(category, kp)
    else:
        name = "acsm_py2pt7_{}_kp_{}_{}".format(category, kp, name)
    parts_file = FLAGS.parts_file
    if category == 'bird':
        if kp:
            command = birds_kp(name, parts_file)
        else:
            command = birds_no_kp(name, parts_file)
    if category == 'horse':
        if kp:
            command = horse_kp(name, parts_file)
        else:
            command = horse_no_kp(name, parts_file)
    else:
        command = category_no_kp(
            category=category,
            name=name,
            parts_file=parts_file,
        )

    print(command)
    # elif category == 'horse':
    #     if with_kp:
    #         command = birds_kp(name, parts_file)

    header = get_sbatch_header(name)
    script_name = "/home/nileshk/DeformParts/sbatch_scripts/{}.sh".format(name)

    with open(script_name, 'w') as f:
        f.write(header)
        f.write('\n')
        f.write('srun {} \n'.format(command))

    print(script_name)
    print(
        "/home/nileshk/DeformParts/refactor/dcsm/cachedir/slurm_logs/{}.out".
        format(name)
    )


if __name__ == '__main__':
    app.run(main)
