# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pdb
import numpy as np
import scipy.io as sio
from point_rend import SemSegDatasetMapper, add_pointrend_config
from detectron2.engine import default_setup
from predictor import VisualizationDemo
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
sys.path.append('/Pool3/users/nileshk/detectron2/demo')


# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def convert_mask_to_bbox(seg_mask):
    from pycocotools import mask
    fortran_mask = np.asfortranarray(seg_mask)
    encoded_mask = mask.encode(fortran_mask)
    mask_area = mask.area(encoded_mask)
    mask_bounding_box = mask.toBbox(encoded_mask)
    return mask_bounding_box


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            if os.path.isdir(args.input[0]):
                args.input = glob.glob(
                    os.path.expanduser(args.input[0] + '*JPEG'))
            else:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(
                        args.output, os.path.basename(path)+'.mat')

            if os.path.exists(out_filename):
                continue
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            instances = predictions['instances'].to(demo.cpu_device)
            class_labels = demo.metadata.get("thing_classes")

            boxes, segms, classes, scores = [], [], [], []
            for ix in range(len(instances)):
                mask = instances.pred_masks[ix].numpy()
                segms.append(mask)
                bbox = convert_mask_to_bbox(mask)
                boxes.append(bbox)
                classes.append(class_labels[instances.pred_classes[ix].item()])
                scores.append(instances.scores[ix].item())

            if len(boxes) == 0:
                continue

            boxes = np.stack(boxes)
            segms = np.stack(segms, axis=2)
            classes = np.asarray(
                ['quad' if 'person' in c else c for c in classes])

            valid_inds = np.array(
                [sx for (sx, s) in enumerate(scores) if s > 0.5])
            if len(valid_inds) == 0:
                continue
            segms = segms[:, :, valid_inds]
            boxes = boxes[valid_inds, :]
            classes = classes[valid_inds]

            sio.savemat(out_filename, {'masks': segms,
                                       'boxes': boxes, 'classes': classes})

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(
                        len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(
                        args.output, os.path.basename(path))
                else:
                    assert len(
                        args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                import pdb
                pdb.set_trace()
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
