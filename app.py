from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import utils 
import registers
from tempfile import NamedTemporaryFile
import mmcv
import cv2

#@st.cache
@st.cache_resource
def register():
    registers.registerstuff()

def process_image(path_img, image, bar):
    config = Path("./20230928_012708.py")
    pathfile = Path("epoch_800.pth")
    model = utils.mmyolo_init(config=config,pathfile=pathfile)
    predictor = utils.sam_init()
    bar.progress(10)
    # -- Inference detection -- #
    detections = utils.inference_detections(model=model, image=path_img) 
    bar.progress(30)
    # -- process inference to inputs for SAM -- #
    inputs_boxes = utils.input_boxes_sam(detections)
    bar.progress(50)
    # -- get prediction information from SAM -- #
    masks_list = utils.prediction_masks_sam(image=image, predictor=predictor, inputs_boxes=inputs_boxes)
    bar.progress(70)
    # -- process these masks into one image array -- #
    batched_mask = utils.masks_array_sam(masks_list=masks_list)
    bar.progress(90)
    # -- return the mask -- #
    return detections, batched_mask


def main():
    register()
    st.sidebar.title("Pipeline: Object Detection -> Semantic Segmentation")
    st.sidebar.divider()
    st.sidebar.header("Upload nuclei image:")
    # -- [ Disable uploader once upload has been done] -- #
    if 'is_uploaded' not in st.session_state:
        st.session_state.is_uploaded = False
    uploaded_image = st.sidebar.file_uploader("Upload H&E stained image (png)", type=["png"], disabled=st.session_state.is_uploaded)
    st.sidebar.divider()
    sidebar_options = st.sidebar
    sidebar_options.markdown("<h1 style='text-align: center; font-size: 40px'>Options</h1>", unsafe_allow_html=True)
    subheader_text = "Please Process Image for Options"
    sidebar_option_subheader = sidebar_options.subheader(subheader_text)
    
    if uploaded_image is not None:
        with NamedTemporaryFile(dir='.', suffix='.png') as f:
            f.write(uploaded_image.getbuffer())
            img = mmcv.imread(f.name)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
            if not st.session_state.is_uploaded:
                st.session_state.is_uploaded = True
                st.rerun()
            processed_image = st.empty()
            # -- Define image for session state -- #
            if 'image' not in st.session_state:
                st.session_state.image = img
            # -- ------------------------------ -- #
            processed_image.image(st.session_state.image)

            # -- Check if button has already been pressed -- #
            if "is_processed" not in st.session_state:
                st.session_state.is_processed = False
            # -- ---------------------------------------- -- #
            cols = st.columns(3)
            if cols[0].button('Process Image'):
                    if cols[1].button('Stop Processing'):
                        st.stop()
                    with st.spinner(text='In progress'):
                        bar = st.progress(0)
                        detections, processed_mask = process_image(f.name, img, bar)
                        if 'detections' not in st.session_state:
                             st.session_state.detections = detections
                        #processed_mask = img
                        bar.progress(100)
                        st.success('Done')
                        st.session_state.is_processed = True
                    
            if st.session_state.is_processed:
                sidebar_option_subheader.subheader("Please choose one of the following options:")
                bounding_box_checkbox = sidebar_options.checkbox("Show Bounding Box", value=False)
                show_mask_checkbox = sidebar_options.checkbox("Show Mask", value=True)

                # -- Process the mask and output the overlay -- #
                if "mask_img" not in st.session_state:
                    total_image = img.copy()
                    processed_mask = np.asarray(processed_mask) * 255
                    processed_mask = cv2.cvtColor(processed_mask.astype(np.float32), cv2.COLOR_RGBA2RGB)
                    total_image_covered = cv2.bitwise_or(total_image, processed_mask.astype(np.uint8))
                    # -- Saving mask + img for future ref -- #
                    st.session_state.mask_img = total_image_covered
                # -- -------------------------------- -- #
                if bounding_box_checkbox and show_mask_checkbox:
                        # -- Show both bounding box and mask on image
                        mask_bound_img = utils.show_box_cv(st.session_state.detections, st.session_state.mask_img.copy())
                        processed_image.image(mask_bound_img)
                if bounding_box_checkbox and not show_mask_checkbox:
                        # -- show only bounding box on original image
                        orig_bound_img = utils.show_box_cv(st.session_state.detections, img.copy())
                        processed_image.image(orig_bound_img)
                if not bounding_box_checkbox and show_mask_checkbox:
                        # -- show only mask on original image
                        processed_image.image(st.session_state.mask_img)
                if not bounding_box_checkbox and not show_mask_checkbox:
                        # -- Just show original image
                        processed_image.image(img)
    else:
        if st.session_state.is_uploaded:
            st.session_state.clear()
            st.rerun()
        print("[*] Clearing all variables from session_state as image has been deleted ")
        

            
            
if __name__ == "__main__":
    main()