import os

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import cv2
import numpy as np

from tqdm import tqdm
import cv2

import argparse
import json
import os
from typing import Any, Dict, List

import time
from tqdm import tqdm


MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


def format_text(item_path):
    item_path = item_path.replace("\\", "/")
    return item_path


def concat_mask_2_npy(item_path, c, h, w):
    npy = np.zeros((c, h, w), dtype=np.uint8)    # shape: (64, 64, 64)
    for idx, img_name in enumerate(os.listdir(item_path)):
        if idx == 64:
            break
        if img_name.endswith('.png'):
            msk = cv2.imread(os.path.join(item_path, img_name), 0)
            npy[idx] = msk
    return npy


def destructor_path(item_path):      # F:\project_AD\MVTec_seg\transistor\test\bent_lead\000
    item_path = format_text(item_path)
    file_name_split_list = item_path.split('/')
    class_name, phase, img_type, item_name = [f for f in file_name_split_list[-4:]]
    return class_name, phase, img_type, item_name


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str, scale_down: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        if scale_down == 'none':
            cv2.imwrite(os.path.join(path, filename), mask * 255)
        elif scale_down == '64':
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (64, 64))
            cv2.imwrite(os.path.join(path, filename), mask)
        else:
            cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def on_segment_branch(dataset_dir, args: argparse.Namespace) -> None:
    model_type = 'vit_b'  # "The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']"
    # 'sam_vit_b_01ec64.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_h_4b8939.pth'
    checkpoint = "F:/project_AD/FF+WT/SAM/sam_vit_b_01ec64.pth"
    device = 'cuda'
    scale_down = '64'  # "The type of model to load, in ['none', '64']"
    phases = ['train', 'test']
    save_dir = dataset_dir + '_seg'

    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    _ = sam.to(device=device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    scale_down = scale_down

    for class_name in os.listdir(dataset_dir):
        if class_name == 'witness_mark':
            continue
        if not os.path.isdir(os.path.join(dataset_dir, class_name)):
            continue
        for phase in phases:
            img_types = os.listdir(os.path.join(dataset_dir, class_name, phase))
            for img_type in img_types:
                input = os.path.join(dataset_dir, class_name, phase, img_type)
                output = os.path.join(save_dir, class_name, phase, img_type)

                if not os.path.isdir(input):
                    targets = [input]
                else:
                    targets = [
                        f for f in os.listdir(input) if not os.path.isdir(os.path.join(input, f))
                    ]
                    targets = [os.path.join(input, f) for f in targets]

                os.makedirs(output, exist_ok=True)

                bar = tqdm(targets)
                for idx, t in enumerate(bar):
                    # print(f"Processing '{t}'...")
                    t1 = time.time()
                    image = cv2.imread(t)
                    if image is None:
                        print(f"Could not load '{t}' as an image, skipping...")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    masks = generator.generate(image)

                    base = os.path.basename(t)
                    base = os.path.splitext(base)[0]
                    save_base = os.path.join(output, base)
                    if output_mode == "binary_mask":
                        os.makedirs(save_base, exist_ok=False)
                        write_masks_to_folder(masks, save_base, scale_down)
                    else:
                        save_file = save_base + ".json"
                        with open(save_file, "w") as f:
                            json.dump(masks, f)
                    t2 = time.time()
                    tt = round(t2 - t1, 2)
                    bar.set_description(f"Processing '{t}', time in {tt}")
                print("Done!")


def on_masks2npy_branch(dataset_path, save_dir, dataset_name):
    class_name_list = []
    if dataset_name == 'mvtec':
        class_name_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # elif dataset_name == 'miad':
    #     class_name_list = ['catenary_dropper', 'electrical_insulator', 'metal_welding',
    #                         'nut_and_bolt', 'photovoltaic_module', 'wind_turbine', 'witness_mark']

    elif dataset_name == 'miad':
        class_name_list = ['catenary_dropper', 'electrical_insulator', 'metal_welding',
                           'nut_and_bolt', 'photovoltaic_module', 'wind_turbine']

    phase_list = ['train', 'test']
    item_list = []

    for class_name in class_name_list:
        # if not class_name == 'transistor':
        #     continue
        for phase in phase_list:
            img_dir = os.path.join(dataset_path, class_name, phase)
            img_types = sorted(os.listdir(img_dir))

            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(img_dir, img_type)  # 'good', ...
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)])
                item_list.extend(img_fpath_list)

    for item_path in item_list:
        class_name, phase, img_type, item_name = destructor_path(item_path)   # 解析路径中的各个文件夹名
        item_name = item_name + '.npy'
        npy = concat_mask_2_npy(item_path, 64, 64, 64)
        if not os.path.exists(os.path.join(save_dir, class_name, phase, img_type)):
            os.makedirs(os.path.join(save_dir, class_name, phase, img_type))
        save_path = os.path.join(save_dir, class_name, phase, img_type, item_name)
        np.save(save_path, npy)

    return 0


if __name__ == '__main__':
    # 使用SAM生成seg mask
    dataset_dir = 'F:/project_AD/MIAD'
    args = parser.parse_args()
    on_segment_branch(dataset_dir, args)

    # 将seg_mask 保存成npy
    dataset_path = dataset_dir + '_seg'
    save_dir     = dataset_dir + '_seg_npy'
    dataset_name = 'miad'
    on_masks2npy_branch(dataset_path, save_dir, dataset_name)

