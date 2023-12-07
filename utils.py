# -- [ MMDETECTION | MMYOLO  Dependencies ] -- #
from pathlib import Path
import pandas as pd
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmseg.structures import SegDataSample
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
import glob
import os
# -- [ For metrics (semantic seg demo) -- ]
import utils_metrics as metrics
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

def show_box_cv(box_s, img):
    for box in box_s:
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        cv2.rectangle(img, (x1,y1), (x2,y2), color=(255,0,0), thickness=1)
    return img

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
    sam_checkpoint = "models/sam/sam_vit_h_4b8939.pth"
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

def numpy_from_result(result: SegDataSample, squeeze: bool = True, as_uint: bool = True) -> np.ndarray:
    """Converts an mmsegmentation inference result into a numpy array (for exporting and visualisation)

    Parameters
    ----------
    result : SegDataSample
        The segmentation results to extract the numpy array from
    squeeze : bool, optional
        Squeezes down useless dimensions (mainly for binary segmentation), by default True
    as_uint : bool, optional
        Converts the array to uint8 format, instead of (usually) int64, by default True

    Returns
    -------
    np.ndarray
        The extracted numpy array
    """    
    array: np.ndarray = result.pred_sem_seg.cpu().numpy().data
    if squeeze:
        array = array.squeeze()
    if as_uint:
        array = array.astype(np.uint8)
    return array

# -- [ Adding more util functions from dataset_selector ] -- #
@st.cache_data
def load_images():
    image_files = glob.glob("images/MoNuSeg/*/*.png")
    images_subset = []
    for image_file in image_files:
        image_file = image_file.replace("\\", "/")
        image_subset = image_file.split("/")
        if image_subset[2] not in images_subset:
            images_subset.append(image_subset[2])

    images_subset.sort()
    return image_files, images_subset

@st.cache_data
def load_images_test_datasets():
    image_files = glob.glob("images/*/test/*.png")
    images_subset = []
    for image_file in image_files:
        image_file = image_file.replace("\\", "/")
        image_subset = image_file.split("/")
        # will select dataset name from list of images
        if image_subset[1] not in images_subset:
            images_subset.append(image_subset[1])
    images_subset.sort()
    return image_files, images_subset

"""
NAME: binary_to_bgr
DESCRIPTION: Function which takes in binary image (GT mask or prediction mask) and 
creates a BGR image of it. This image can be seen and used for displaying.
"""
def binary_to_bgr(img: np.ndarray) -> np.ndarray:
    _,pred_mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
    pred_mask_3d = np.zeros((np.array(pred_mask).shape[0], np.array(pred_mask).shape[1],3),dtype="uint8")   
    pred_mask_3d[:,:,0] = pred_mask
    pred_mask_3d[:,:,1] = pred_mask
    pred_mask_3d[:,:,2] = pred_mask
    return pred_mask_3d

def name_processer(img:str):
    imagepre = img.split(sep="/")
    name = imagepre[3]
    return name

def mask_searcher(name:str):
    mask_file = f"images/MoNuSeg/masks/test/{name}"
    file = None
    if os.path.isfile(mask_file):
        file = mask_file
    else:
        file = None
    return file
"""
NAME: model_accuracy
DESC: Function which gives the metrics for a particular image from prediction and GT
"""
@st.cache_data
def model_accuracy(ground: Path, prediction: Path, class_idx=1) -> pd.DataFrame:
    results = {}
    results["name"] = ground.stem
    # need to read in the data
    gt = cv2.imread(str(ground),cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(prediction),cv2.IMREAD_GRAYSCALE)

    results["accuracy"] = str(metrics.accuracy(gt, pred))
    results["precision"] = str(metrics.precision(gt, pred))
    results["recall"] = str(metrics.recall(gt, pred))
    results["f1"] = str(metrics.f1(metrics.precision(gt, pred), metrics.recall(gt, pred)))
    results["iou"] = str(metrics.iou(gt, pred))
    df = pd.DataFrame.from_dict(results, orient='index',)
    return df

"""
NAME: gt_pred_overlay
DESC: Function which gives a colourful image of how close prediction mask was to the ground truth
"""
@st.cache_data
def gt_pred_overlay(ground: Path, prediction: Path):
    gt = cv2.imread(str(ground), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(prediction), cv2.IMREAD_GRAYSCALE)
    gt_mask_3d = binary_to_bgr(img=gt)
    pred_mask_3d = binary_to_bgr(img=pred)
    # -- [ adding colour to binary images ] -- #
    green = [0, 255, 0]
    blue =  [0, 0, 255]
    yel =   [255, 255, 0]
    red =   [255, 0, 0] 
    white = [255, 255, 255]
    # -- [ Colouring all white pixels to another colour] -- #
    pred_mask_3d[np.where((pred_mask_3d==[255,255,255]).all(axis=2))] = red # Prediction mask
    gt_mask_3d[np.where((gt_mask_3d==[255,255,255]).all(axis=2))] = blue    # Ground truth mask
    # -- [ Converting all image masks made above to RGB ] -- #
    pred_mask_3d = cv2.cvtColor(pred_mask_3d, cv2.COLOR_BGR2RGB)
    gt_mask_3d = cv2.cvtColor(gt_mask_3d, cv2.COLOR_BGR2RGB)
    # -- [ Perform the subtraction ] -- #
    overlap = cv2.bitwise_xor(gt_mask_3d,pred_mask_3d)
    return overlap