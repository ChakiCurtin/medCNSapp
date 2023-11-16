from io import BufferedReader, BytesIO
from mmengine import Config
from mmseg.apis import init_model, inference_model
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

def models_selector(chosen_model:str):
    models_dict = {
        "U-Net": unet_processor,
        "Deeplabv3+": deeplab_processor,
        "MMYOLOv8": mmyolo_processor,
        "Yolov8": yolo_processor,
        "MMYOLO -> SAM": process_image_pipeline,
     }
    return models_dict.get(chosen_model)

def show_selector(chosen_model:str):
    models_dict = {
        "U-Net": unet_show,
        "Deeplabv3+": deeplab_show,
        "MMYOLOv8": mmyolo_show,
        "Yolov8": yolo_show,
        "MMYOLO -> SAM": pipeline_show,
     }
    return models_dict.get(chosen_model)
     
def unet_processor(path_img, image, bar):
    config = Path("./models/semantic/unet/unet_test/unet.py")
    pathfile = Path("./models/semantic/unet/unet_test/iter_20000.pth")
    cfg = Config.fromfile(config)
    bar.progress(20)
    model = init_model(cfg, str(pathfile), 'cuda:0')
    bar.progress(30)
    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']
    bar.progress(40)
    pred = process_images(path_img, model, classes, palette)
    bar.progress(100)
    st.image(pred)

def deeplab_processor(path_img, image, bar):
    config = Path("./models/semantic/deeplab/deeplab_test/deeplab.py")
    pathfile = Path("./models/semantic/deeplab/deeplab_test/iter_20000.pth")
    cfg = Config.fromfile(config)
    bar.progress(20)
    model = init_model(cfg, str(pathfile), 'cuda:0')
    bar.progress(30)
    classes = model.dataset_meta['classes']
    palette = model.dataset_meta['palette']
    bar.progress(40)
    pred = process_images(path_img, model, classes, palette)
    bar.progress(100)
    st.image(pred)

def process_images(path_img, model, classes, palette):
    img = cv2.imread(path_img)
    result = inference_model(model, img)
    mask = utils.numpy_from_result(result=result)
    dest = mask_overlay(img, mask, classes, palette)
    return dest

def mask_overlay(img, mask, classes, palette):
    dest = img.copy()
    labels = np.unique(mask)

    for label in labels:
        # skipping background (ugly in visualisations)
        if classes[label].lower() in ["background", "bg"]:
            continue
        binary_mask = (mask == label)
        colour_mask = np.zeros_like(img)
        # -- Skipping outline for now
        colour_mask[...] = palette[label]
        dest[binary_mask] = cv2.addWeighted(img, 1 - 0.5, colour_mask, 0.5, 0)[binary_mask]
    return dest 

def mmyolo_processor(path_img, image, bar):
    return "mmyolo"

def yolo_processor(path_img, image, bar):
    return "yolo"

def unet_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    return "unet"

def deeplab_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    return "deep"

def mmyolo_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    return "mmyolo"

def yolo_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    return "yolo"

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
    # -- add the mask to current session-- #
    st.session_state.detections = detections
    st.session_state.batched_mask = batched_mask

def pipeline_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    sidebar_option_subheader.subheader("Please choose one of the following options:")
    bounding_box_checkbox = side_tab_options.checkbox("Show Bounding Box", value=False)
    show_mask_checkbox = side_tab_options.checkbox("Show Mask", value=True)

    # -- Process the mask and output the overlay -- #
    if "mask_img" not in st.session_state:
        total_image = og_img.copy()
        batched_mask = np.asarray(st.session_state.batched_mask) * 255
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
            side_tab_settings.warning("The model was trained on the MoNuSeg dataset.")
        elif model_option == "Object Detection":
            overall_model = side_tab_settings.radio(label=" ",
                                                      options=["MMYOLOv8","Yolov8",],
                                                      captions=["OpenMMLab implementation of Yolov8","The original Yolov8"])
            side_tab_settings.divider()
            side_tab_settings.warning("Object detection models do not show mask, only a bounding box around the seperate detected nuceli.")
            side_tab_settings.warning("The model was trained on the MoNuSeg dataset.")
        elif model_option == "Pipeline: Object Detection -> Semantic Segmentation":
            overall_model = side_tab_settings.radio(label=" ",
                                                      options=["MMYOLO -> SAM"],
                                                      captions=["Object detection through MMYOLO with Segment anything model (SAM)"])
            side_tab_settings.divider()
            side_tab_settings.warning("The pipeline will use the chosen object detector for the input bounding boxes which will then be used with the segment anything model (SAM) to produce the mask around the detected nuceli.")
            side_tab_settings.warning("The model was trained on the MoNuSeg dataset.")
        st.session_state.model_option = overall_model

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
                            models_selector(st.session_state.model_option)(f.name, img, bar)
                            bar.progress(100)
                            st.success('Done')
                            st.session_state.is_processed = True
                    else:
                        st.warning("Please Choose a model in the 'Settings' tab on the left (sidebar)")
                        st.stop()
                    
            if st.session_state.is_processed:
                st.session_state.process_button = True
                show_selector(st.session_state.model_option)(processed_image=processed_image,
                                                             side_tab_options=side_tab_options,
                                                             sidebar_option_subheader=sidebar_option_subheader,
                                                             main_col_1=cols[1],
                                                             og_img=img,
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
        if "batched_mask" in st.session_state:
            del st.session_state.batched_mask
            print("[**] cleared batched_mask")
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