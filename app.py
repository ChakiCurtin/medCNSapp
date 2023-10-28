from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import utils 
from registers import registerStuff
from tempfile import NamedTemporaryFile
import mmcv
import cv2

#@st.cache
@st.cache_resource
def register():
    registerStuff()

def process_image(path_img, image, bar):
    bar.progress(0)
    config = Path("./20230928_012708.py")
    pathfile = Path("epoch_800.pth")
    model = utils.mmyolo_init(config=config,pathfile=pathfile)
    predictor = utils.sam_init()
    bar.progress(10)
    # -- Inference detection -- #
    detections = utils.inference_detections(model=model, image=path_img)
    bar.progress(30)
    # -- process inference to inputs for SAM -- #
    inputs_boxes = utils.input_boxes_SAM(detections)
    bar.progress(60)
    # -- get prediction information from SAM -- #
    masks_list = utils.prediction_masks_SAM(image=image, predictor=predictor, inputs_boxes=inputs_boxes)
    bar.progress(80)
    # -- process these masks into one image array -- #
    batched_mask = utils.masks_array_SAM(image=image, masks_list=masks_list)
    bar.progress(95)
    # -- return the mask -- #
    return batched_mask


def main():
    register()
    st.title("Semantic Segmentation using SAM through Object detection model")
    st.header("Upload nuclei image or choose from list below")
    uploaded_image = st.file_uploader("Upload H&E stained image (png)", type=["png"])
    
    if uploaded_image is not None:
        with NamedTemporaryFile(dir='.', suffix='.png') as f:
            f.write(uploaded_image.getbuffer())
            # left_co, cent_co,last_co = st.columns(3)
            # with cent_co:
            #     st.image(data_file)
            img = mmcv.imread(f.name)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            st.image(img)

            if st.button('Process Image'):
                with st.spinner(text='In progress'):
                    alpha: float = 0.5
                    bar = st.progress(0)
                    processed_mask = process_image(f.name, img, bar)
                    bar.progress(100)
                    st.success('Done')
                    total_image = img.copy()
                    total_image = cv2.cvtColor(total_image, cv2.COLOR_RGB2RGBA).copy()
                    st.write(processed_mask.shape)
                    total_image[processed_mask] = cv2.addWeighted(img, 1 -alpha, processed_mask, alpha, 0)[processed_mask]
                    st.image(total_image)
                    st.image(processed_mask)
            
            
if __name__ == "__main__":
    main()