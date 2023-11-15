from io import BufferedReader, BytesIO
from pathlib import Path
import streamlit as st
import numpy as np
import utils 
import registers
from tempfile import NamedTemporaryFile
import cv2

# -- [ Page settings ] -- #
st.set_page_config(page_title="Home | Image Segmenter", 
                   initial_sidebar_state="expanded",
                   layout="centered",
                   menu_items={
                        'About': " # App made by Chaki Ramesh.\n"
                        "Used Machine learning and Computer vision techniques to create a object detection -> instance segmentation (semantic) pipeline"
                        },
                   )
# ################################################################## #
# -- [ Custom CSS STUFF ] -- #
# -- -- [ allowing hover over image enlargment] -- -- #
st.markdown(
    """
    <style>
    img {
        cursor: pointer;
        transition: all .1s ease-in-out;
    }
    img:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# -- [ "Remove the "made with streamlit" at the footer of the page]
hide_streamlit_style = """
            <style>
            MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# ################################################################## #
#@st.cache
@st.cache_resource
def register():
    registers.registerstuff()

# TODO: Add multiple models for processing images. 
# Bounding boxes should still be able to be extracted and added. Semantic segmentation methods like UNET and DEEPLAB and
# Current pipeline methods for mmyolo, rtmdet with SAM should be added
def process_image_pipeline(path_img, image, bar):
    config = Path("./models/objdetection/mmyolo/mmyolov8/mmyolov8_config.py")
    pathfile = Path("./models/objdetection/mmyolo/mmyolov8/epoch_800.pth")
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

def pipeline_processed(batched_mask, processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    sidebar_option_subheader.subheader("Please choose one of the following options:")
    bounding_box_checkbox = side_tab_options.checkbox("Show Bounding Box", value=False)
    show_mask_checkbox = side_tab_options.checkbox("Show Mask", value=True)

    # -- Process the mask and output the overlay -- #
    if "mask_img" not in st.session_state:
        total_image = og_img.copy()
        batched_mask = np.asarray(batched_mask) * 255
        batched_mask = cv2.cvtColor(batched_mask.astype(np.float32), cv2.COLOR_RGBA2RGB)
        total_image_covered = cv2.bitwise_or(total_image, batched_mask.astype(np.uint8))
        # -- Saving mask + img for future ref -- #
        st.session_state.mask_img = total_image_covered
    # -- -------------------------------- -- #
    show_image = og_img
    if bounding_box_checkbox and show_mask_checkbox:
            # -- Show both bounding box and mask on image
            mask_bound_img = utils.show_box_cv(st.session_state.detections, st.session_state.mask_img.copy())
            show_image = mask_bound_img
    if bounding_box_checkbox and not show_mask_checkbox:
            # -- show only bounding box on original image
            orig_bound_img = utils.show_box_cv(st.session_state.detections, og_img.copy())
            show_image = orig_bound_img
    if not bounding_box_checkbox and show_mask_checkbox:
            # -- show only mask on original image
            show_image = st.session_state.mask_img
    if not bounding_box_checkbox and not show_mask_checkbox:
            # -- Just show original image
            show_image = og_img
    processed_image.image(show_image, caption=st.session_state.uploaded_image.name)

    main_col_1.download_button(label="Download", data=download_image(show_image), file_name=st.session_state.uploaded_image.name, mime="image/png")

def main():
    st.sidebar.title("Single Cell Nuclei Segmentation")
    side_tab_settings, side_tab_image_upload, side_tab_options = st.sidebar.tabs(["\u2001\u2001\u2001Settings\u2001\u2001",
                                                                                   "\u2001\u2001Image Upload\u2001\u2001" ,
                                                                                   "\u2001\u2001Options\u2001\u2001\u2001",
                                                                                   ])                                 
    
    # -- [ SETTINGS TAB INFO ] -- #
    side_tab_settings.title("Choose Model:")
    model_option = side_tab_settings.selectbox(label="Choose Model Range",
                                options=('Semantic Segmentation', 'Object Detection', 'Pipeline: Object Detection -> Semantic Segmentation',),
                                index=None,
                                placeholder="Choose a Model Type",
                                )
    if model_option is not None:
        overall_model = None
        if model_option == "Semantic Segmentation":
            overall_model = side_tab_settings.radio(label=" ",
                                                      options=["U-Net","Deeplabv3+",],
                                                      captions=["Popular in medical research","Newer and and upgrade over U-Net"])
            side_tab_settings.divider()
            side_tab_settings.warning("Semantic segmentation models do not show bounding boxes around the seperate nuclei, it will only show mask on image.")
        elif model_option == "Object Detection":
            overall_model = side_tab_settings.radio(label=" ",
                                                      options=["MMYOLOv8","Yolov8",],
                                                      captions=["OpenMMLab implementation of Yolov8","The original Yolov8"])
            side_tab_settings.divider()
            side_tab_settings.warning("Object detection models do not show mask, only a bounding box around the seperate detected nuceli.")

        elif model_option == "Pipeline: Object Detection -> Semantic Segmentation":
            overall_model = side_tab_settings.radio(label=" ",
                                                      options=["MMYOLO -> SAM"],
                                                      captions=["Object detection through MMYOLO with Segment anything model (SAM)"])
            side_tab_settings.divider()
            side_tab_settings.warning("The pipeline will use the chosen object detector for the input bounding boxes which will then be used with the segment anything model (SAM) to produce the mask around the detected nuceli.")

             
        st.session_state.model_option = model_option

    # -- [ IMAGE UPLOAD TAB INFO ] -- #
    side_tab_image_upload.header("Upload nuclei image:")
    # -- [ Disable uploader once upload has been done] -- #
    if 'is_uploaded' not in st.session_state:
        st.session_state.is_uploaded = False
    uploaded_image = side_tab_image_upload.file_uploader("Upload H&E stained image (png)", type=["png"], disabled=st.session_state.is_uploaded)
    # TODO - Add multi image input (list of images for processing)

    # -- [ OPTIONS TAB INFO ] -- #
    side_tab_options.markdown("<h1 style='text-align: center; font-size: 40px'>Options</h1>", unsafe_allow_html=True)
    subheader_text = "Please Process Image for Options"
    sidebar_option_subheader = side_tab_options.subheader(subheader_text)


    if uploaded_image is not None:
        if 'uploaded_image' not in st.session_state:
             st.session_state.uploaded_image = uploaded_image
        with NamedTemporaryFile(dir='.', suffix='.png') as f:
            f.write(uploaded_image.getbuffer())
            img = cv2.imread(f.name)
            if not st.session_state.is_uploaded:
                st.session_state.is_uploaded = True
                st.rerun()
            processed_image = st.empty()
            # -- Define image for session state -- #
            if 'image' not in st.session_state:
                st.session_state.image = img
            # -- ------------------------------ -- #
            processed_image.image(st.session_state.image, caption=uploaded_image.name)

            # -- Check if button has already been pressed -- #
            if "is_processed" not in st.session_state:
                st.session_state.is_processed = False
            # -- ---------------------------------------- -- #
            cols = st.columns(3)
            if 'process_button' not in st.session_state:
                 st.session_state.process_button = False
            if cols[0].button('Process Image', disabled=st.session_state.process_button):
                    if "model_option" in st.session_state:
                        if cols[1].button('Stop Processing', disabled=st.session_state.process_button):
                            st.stop()
                        with st.spinner(text='In progress'):
                            bar = st.progress(0)
                            detections, processed_mask = process_image_pipeline(f.name, img, bar)
                            if 'detections' not in st.session_state:
                                st.session_state.detections = detections
                            if "processed_mask" not in st.session_state:
                                 st.session_state.processed_mask = processed_mask
                            bar.progress(100)
                            st.success('Done')
                            st.session_state.is_processed = True
                    else:
                        st.warning("Please Choose a model in the 'Settings' tab on the left (sidebar)")
                        st.stop()
                    
            if st.session_state.is_processed:
                st.session_state.process_button = True
                # sidebar_option_subheader.subheader("Please choose one of the following options:")
                # bounding_box_checkbox = side_tab_options.checkbox("Show Bounding Box", value=False)
                # show_mask_checkbox = side_tab_options.checkbox("Show Mask", value=True)

                # # -- Process the mask and output the overlay -- #
                # if "mask_img" not in st.session_state:
                #     total_image = img.copy()
                #     processed_mask = np.asarray(processed_mask) * 255
                #     processed_mask = cv2.cvtColor(processed_mask.astype(np.float32), cv2.COLOR_RGBA2RGB)
                #     total_image_covered = cv2.bitwise_or(total_image, processed_mask.astype(np.uint8))
                #     # -- Saving mask + img for future ref -- #
                #     st.session_state.mask_img = total_image_covered
                # # -- -------------------------------- -- #
                # show_image = img
                # if bounding_box_checkbox and show_mask_checkbox:
                #         # -- Show both bounding box and mask on image
                #         mask_bound_img = utils.show_box_cv(st.session_state.detections, st.session_state.mask_img.copy())
                #         show_image = mask_bound_img
                # if bounding_box_checkbox and not show_mask_checkbox:
                #         # -- show only bounding box on original image
                #         orig_bound_img = utils.show_box_cv(st.session_state.detections, img.copy())
                #         show_image = orig_bound_img
                # if not bounding_box_checkbox and show_mask_checkbox:
                #         # -- show only mask on original image
                #         show_image = st.session_state.mask_img
                # if not bounding_box_checkbox and not show_mask_checkbox:
                #         # -- Just show original image
                #         show_image = img
                # processed_image.image(show_image, caption=st.session_state.uploaded_image.name)

                # cols[1].download_button(label="Download", data=download_image(show_image), file_name=st.session_state.uploaded_image.name, mime="image/png")
                pipeline_processed(batched_mask=st.session_state.processed_mask, 
                                   main_col_1=cols[1],
                                   og_img=img,
                                   processed_image=processed_image,
                                   side_tab_options=side_tab_options,
                                   sidebar_option_subheader=sidebar_option_subheader,
                                   )
    else:
        st.warning("Please use the sidebar <- to choose the model, upload the image for processing.")
        print("[*] Clearing local variables stored in cache for the image")
        if 'uploaded_image' in st.session_state:
             del st.session_state.uploaded_image
             print("[**] cleared uploaded_image")
        if 'image' in st.session_state:
            del st.session_state.image
            print("[**] cleared image")
        if "is_processed" in st.session_state:
            del st.session_state.is_processed
            print("[**] cleared is_processed")
        if "process_button" in st.session_state:
            del st.session_state.process_button
            print("[**] cleared process_button")
        if 'detections' in st.session_state:
            del st.session_state.detections
            print("[**] cleared detections")
        if "mask_img" in st.session_state:
            del st.session_state.mask_img
            print("[**] cleared mask_img")
        if "processed_mask" in st.session_state:
            del st.session_state.processed_mask
        if st.session_state.is_uploaded:
            del st.session_state.is_uploaded
            print("[**] cleared is_uploaded")
            st.rerun()


def download_image(img):
    img = img[:, :, [2, 1, 0]] #numpy.ndarray # from bgr to rgb
    _, img_enco = cv2.imencode(".png", img)
    srt_enco = img_enco.tobytes() #bytes
    img_bytesio = BytesIO(srt_enco) #_io.BytesIO
    img_bufferedreader = BufferedReader(img_bytesio) #_io.BufferedReader
    return img_bufferedreader


            
if __name__ == "__main__":
    register()
    main()