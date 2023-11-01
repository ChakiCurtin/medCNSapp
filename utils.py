# -- [ MMDETECTION | MMYOLO  Dependencies ] -- #
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import argparse 
import sys
import numpy as np
import gc
import matplotlib.pyplot as plt
import cv2
# -- [ SEGMENT ANYTHING MODEL ] -- #
from segment_anything import sam_model_registry, SamPredictor
# -- [ STREAMLIT TOOLS ] -- #
import streamlit as st

# -- ############################### -- #

# -- [ Show Mask Function ] -- #
'''
INPUT: Mask: ? , ax: plt.gca() , random_colour: bool 
OUTPUT: Null 
DEF: Function takes the mask produced by SAM, reshapes it and adds colours and then adds to plot
'''
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.Generator.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask_image
    gc.collect()

# -- [ Show Box Function ] -- #
'''
INPUT: box: ?, ax: plt.gca()
OUTPUT: Null
DEF: Function takes bounding box produced and add it to plot
'''
def show_box_plt(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1))    

# -- [ Get images function ] -- #
def get_images_old(args: argparse.Namespace):
    images = args.dataset_root / args.dataset / args.set
    return sorted(images.glob("*.png"))


# -- [ YOLO INIT ] -- #
@st.cache_resource
def mmyolo_init(config, pathfile):
    print("[*] Loading path file...")
    print("[*] Loading config file...")
    print("[*] Building model..")
    model = init_detector(str(config), str(pathfile), device='cuda:0')
    print("[*] Initialising Visualiser..")
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model

# -- [SAM INIT ] -- #
@st.cache_resource
def sam_init():
    sys.path.append("..")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def inference_detections(model, image):
    #print("[*] Generating result for image: " + image.name)
    result = inference_detector(model, image)
    nuclei_list = []
    for xx in range(1000):
        X1 = int(result.pred_instances.bboxes[xx][0])
        Y1 = int(result.pred_instances.bboxes[xx][1])
        X2 = int(result.pred_instances.bboxes[xx][2])
        Y2 = int(result.pred_instances.bboxes[xx][3])

        nuclei_list.append([X1,Y1,X2,Y2])
    return nuclei_list

@st.cache_data
def input_boxes_sam(nuclei_list):
    inputs_boxes = [nuclei_list[0:199],nuclei_list[200:399],nuclei_list[400:599],nuclei_list[600:799],nuclei_list[800:999]]
    return inputs_boxes


def prediction_masks_sam(image, predictor, inputs_boxes):
    masks_list = []
    predictor.set_image(image)
    for section in inputs_boxes:
        input_box = np.array(section)
        input_box = torch.tensor(input_box, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        masks_list.append(masks)
    return masks_list


def masks_array_sam(masks_list):
    masked_out_list = []
    for masks in masks_list:
        #print("[**] Plotting batch masks | batch: " + str(ii))
        sub_mask_list = []
        for mask in masks:
            #show_mask(mask.cpu().numpy(), plt.gca())
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.cpu().numpy().shape[-2:]
            mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
            sub_mask_list.append(mask_image)
        #print("[**] saving batch of masks | batch: " + str(ii))
        batched_mask = sub_mask_list[0]
        for iii in range(len(sub_mask_list)):
            batched_mask = cv2.bitwise_or(batched_mask, sub_mask_list[iii])
        masked_out_list.append(batched_mask)
    
    batched_mask = masked_out_list[0]
    for ii in range(len(masked_out_list)):
        batched_mask = cv2.bitwise_or(batched_mask, masked_out_list[ii])
    
    return batched_mask

