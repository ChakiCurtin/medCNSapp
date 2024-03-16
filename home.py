import base64
from io import BufferedReader, BytesIO, StringIO
import io
import os
# -- [ For OpenMMLab (U-Net, DeepLabv3+, mmyolov8,  )(MMDETECTION, MMSEGMENTATION, MMYOLO) ] -- #
from mmengine import Config
from mmseg.apis import init_model, inference_model
import registers
# -- [ For yolov8 ] -- #
# from ultralytics import YOLO
# -- [ The rest of the imports ] -- #
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import utils 
from tempfile import NamedTemporaryFile
import cv2
import plotly.express as px
# -0- testing new python package -- #
from st_clickable_images import clickable_images
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


# TODO[very low]: Get more models to use for selection. RTMDet, fasterSAM etc
def models_selector(chosen_model:str):
    models_dict = {
        "U-Net": unet_processor,
        "Deeplabv3+": deeplab_processor,
        "MMYOLOv8": mmyolo_processor,
        #"Yolov8": yolo_processor,
        "MMYOLO -> SAM": process_image_pipeline,
        "custom": custom_processor,
     }
    return models_dict.get(chosen_model)


# TODO[low/medium]: Create a class for the overlap and test
def show_selector(chosen_model:str):
    models_dict = {
        "U-Net": semantic_show,
        "Deeplabv3+": semantic_show,
        "MMYOLOv8": instance_seg_show,
        "Yolov8": instance_seg_show,
        "MMYOLO -> SAM": pipeline_show,
        "custom": semantic_show
     }
    return models_dict.get(chosen_model)
     
# TODO[very low]: refactor this and other processors to work for all
def custom_processor(path_img, image, bar):
    cfg = None
    model = None
    if "custom_model" in st.session_state:
        if st.session_state.custom_model != None:
            with NamedTemporaryFile(dir="./temp/", suffix=".py",delete=True) as f:
                f.write(st.session_state.custom_model.getvalue())
                cfg = Config.fromfile(f.name)
                f.close()
        else:
            st.warning("Custom Model was not given, please upload the custom model config file")
            st.stop()
    else:
        st.warning("Custom Model was not given, please upload the custom model config file")
        st.stop()
    if "custom_weights" in st.session_state:
        if st.session_state.custom_weights != None:
            # this should take the uploaded file, create the temp .pth file and sent that to the model_init
            with NamedTemporaryFile(dir="./temp/", suffix=".pth",delete=True) as f:
                f.write(st.session_state.custom_weights.getvalue())
                if cfg.default_scope == "mmseg":
                    model = init_model(cfg, str(f.name), 'cuda:0')
                # TODO[low]: Support MMDET and MMYOLO 
                # elif cfg.default_scope == "mmyolo":
                # model = utils.mmyolo_init(config=,pathfile=str(f.name))
                f.close()
        else:
            st.warning("Custom Weights was not given, please upload the custom model weights file")
            st.stop()
    else:
        st.warning("Custom Weights was not given, please upload the custom model weights file")
        st.stop()

    if cfg != None and model != None:
        st.session_state.model_chosen = f"Custom | Default Scope: {cfg.default_scope}"
        bar.progress(30)
        classes = model.dataset_meta['classes']
        palette = [[0,0,0],[0,255,0]]
        bar.progress(40)
        pred = process_images(path_img, model, classes, palette)
        bar.progress(100)
        # -- add the mask to current session-- #
        st.session_state.batched_mask = pred


def unet_processor(path_img, image, bar):
    st.session_state.model_chosen = "U-Net"
    config = Path("./models/semantic/unet/unet_test/unet.py")
    pathfile = Path("./models/semantic/unet/unet_test/iter_20000.pth")
    if not Path.exists(pathfile):
        with st.spinner("Dowloading model weights..."):
            print("[*] Downloading unet weights from gdrive")
            utils.download_path(modelstr="unet")
    cfg = Config.fromfile(config)
    bar.progress(20)
    model = init_model(cfg, str(pathfile), 'cuda:0')
    bar.progress(30)
    classes = model.dataset_meta['classes']
    palette = [[0,0,0],[0,255,0]]
    bar.progress(40)
    pred = process_images(path_img, model, classes, palette)
    bar.progress(100)
    # -- add the mask to current session-- #
    st.session_state.batched_mask = pred

def deeplab_processor(path_img, image, bar):
    st.session_state.model_chosen = "Deeplabv3+"
    config = Path("./models/semantic/deeplab/deeplab_test/deeplab.py")
    pathfile = Path("./models/semantic/deeplab/deeplab_test/iter_20000.pth")
    if not Path.exists(pathfile):
        print("[*] Downloading deeplabv3+ weights from gdrive")
        with st.spinner("Dowloading model weights..."):
            utils.download_path(modelstr="deeplab")
    cfg = Config.fromfile(config)
    bar.progress(20)
    model = init_model(cfg, str(pathfile), 'cuda:0')
    bar.progress(30)
    classes = model.dataset_meta['classes']
    palette = [[0,0,0],[0,255,0]]
    bar.progress(40)
    pred = process_images(path_img, model, classes, palette)
    bar.progress(100)
    # -- add the mask to current session-- #
    st.session_state.batched_mask = pred

def process_images(path_img, model, classes, palette):
    img = cv2.imread(path_img)
    result = inference_model(model, img)
    mask = utils.numpy_from_result(result=result) # Prediction raw
    # - - [ Saving raw image to session state to be used for different applications ] - - #
    st.session_state.pred_mask_raw = mask
    dest = mask_overlay(img, mask, classes, palette)
    # TODO[low]: Add metrics and overlay and pred_mask_raw to get deleted on image reset
    # For now use the fact that sample is only created if it is chosen as a sample image
    # for checking.
    if ("sample" in st.session_state) or ("gt_mask" in st.session_state):
        if st.session_state.sample:
            metrics = generate_metrics_per_img(path_img)
            st.session_state.metrics = metrics
        elif "gt_mask" in st.session_state:
            print("[*] HOME.PY | process_images | gt_mask provided")
            gt_img = str(st.session_state.gt_mask)
            metrics = generate_metrics_per_img(gt_img)
            st.session_state.metrics = metrics
    return dest

def generate_metrics_per_img(img_path:str):
    gt_path = img_path
    if "sample" in st.session_state:
        if st.session_state.sample:
            gt_path = utils.name_processer(img=img_path)
            gt_path = utils.mask_searcher(gt_path)
        else:
            if "gt_mask" in st.session_state:
                gt_path = img_path
    print(f"File for metrics: {gt_path}")
    raw_mask:np.ndarray = st.session_state.pred_mask_raw # Prediction raw mask
    f = NamedTemporaryFile(dir='./temp', suffix='.png', delete=True)
    _, buffer = cv2.imencode(".png", raw_mask) # Required for encoding binary images to file
    io_buf = io.BytesIO(buffer) # Create a bufferable encoded file
    f.write(io_buf.getbuffer()) # write to file
    df = utils.model_accuracy(ground=Path(gt_path),prediction=Path(f.name))
    # -- [ Get overlay for both gt and pred ] -- #
    overlay =  utils.gt_pred_overlay(ground=Path(gt_path),prediction=Path(f.name))
    st.session_state.overlay = overlay
    return df


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
        # TODO[medium/low]: Make transparency either a slider or modifiable.
        colour_mask[...] = palette[label]
        dest[binary_mask] = cv2.addWeighted(img, 1 - 0.5, colour_mask, 0.5, 0)[binary_mask]
    return dest 

# TODO[low]: Create Yolo processor for ultralytics
# def yolo_processor(path_img, image, bar):
#     st.session_state.model_chosen = "Yolov8"
#     pathfile = "models/yolo/150_epoch_batch_2/yolo_ultra_150_epochs/train3/weights/best.pt"
#     model=YOLO(pathfile)
#     return "yolo"


def mmyolo_processor(path_img, image, bar):
    # TODO[high]: Split the pipeline option to allow users to select the obj detector which parses info into SAM. (done in pipeline_extractor)
    st.session_state.model_chosen = "MMYolov8"
    config = Path("./models/objdetection/mmyolo/mmyolov8/mmyolov8_config.py")
    pathfile = Path("./models/objdetection/mmyolo/mmyolov8/epoch_800.pth")
    if not Path.exists(pathfile):
        print("[*] Downloading mmyolov8 weights from gdrive")
        utils.download_path(modelstr="mmyolo")
    model = utils.mmyolo_init(config=config,pathfile=pathfile)
    # -- Inference detection -- #
    detections = utils.inference_detections(model=model, image=path_img)
    orig_bound_img = utils.show_box_cv(detections, image.copy())
    st.session_state.bounded_img = orig_bound_img

def instance_seg_show(processed_image, og_img, sidebar_option_subheader, side_tab_options, main_col_1):
    # TODO[low/medium]: Show GT bounding box regions for sample images as an overlay option
    # TODO[low]: Be able to change colour of bounding box regions based on user preference
    sidebar_option_subheader.subheader("Please choose one of the following options:")
    if "bound_check" not in st.session_state:
        st.session_state.bound_check = True
    if "image_check" not in st.session_state:
        st.session_state.image_check = True
    #if "overlay_check" not in st.session_state:
    #    st.session_state.overlay_check = False
    show_bound_checkbox = side_tab_options[2].checkbox("Show detections", value=st.session_state.bound_check)
    show_image_checkbox = side_tab_options[2].checkbox("Show Image", value=st.session_state.image_check)
    # if "sample" in st.session_state:
    #     if st.session_state.sample:
    #         show_overlay_checkbox = side_tab_options[2].checkbox("Show Overlay", value= st.session_state.overlay_check)
    #     else:
    #         show_overlay_checkbox = False
    # else:
    #     show_overlay_checkbox = False
    
    # -- [ setting all checkboxes]

    if show_bound_checkbox and show_image_checkbox:
        # show both
        show_image = st.session_state.bounded_img
    elif not show_bound_checkbox and show_image_checkbox:
        # just show original image
        show_image = og_img

    if not show_bound_checkbox and not show_image_checkbox:
        processed_image.empty()
    elif show_bound_checkbox and not show_image_checkbox:
        processed_image.empty()
        st.warning("Cannot should bounding box region without image. Please choose both.")
    else:
        fig = px.imshow(show_image,height=800,aspect='equal')
        processed_image.plotly_chart(fig,use_container_width=True)
    

def semantic_show(processed_image, og_img, sidebar_option_subheader, side_tab_options, main_col_1):
    sidebar_option_subheader.subheader("Please choose one of the following options:")
    if "mask_check" not in st.session_state:
        st.session_state.mask_check = True
    if "image_check" not in st.session_state:
        st.session_state.image_check = True
    if "overlay_check" not in st.session_state:
        st.session_state.overlay_check = False
    show_mask_checkbox = side_tab_options[2].checkbox("Show Mask", value=st.session_state.mask_check)
    show_image_checkbox = side_tab_options[2].checkbox("Show Image", value=st.session_state.image_check)
    if "sample" in st.session_state or "gt_mask" in st.session_state:
        if st.session_state.sample:
            show_overlay_checkbox = side_tab_options[2].checkbox("Show Overlay", value= st.session_state.overlay_check)
        elif "gt_mask" in st.session_state:
            show_overlay_checkbox = side_tab_options[2].checkbox("Show Overlay", value= st.session_state.overlay_check)
        else:
            show_overlay_checkbox = False
    else:
        show_overlay_checkbox = False
    
    # -- [ setting all checkboxes]
    OVERLAY_WARNING="Overlay option has to be the only option toggled. This will show overlay only"
    if show_mask_checkbox and not show_image_checkbox and not show_overlay_checkbox:
        show_image = utils.binary_to_bgr(img=st.session_state.pred_mask_raw)
        fig = px.imshow(show_image,height=800,aspect='equal')
    elif not show_mask_checkbox and show_image_checkbox and not show_overlay_checkbox:
        show_image = og_img
        fig = px.imshow(show_image,height=800,aspect='equal')
    elif not show_mask_checkbox and not show_image_checkbox and show_overlay_checkbox:
        show_image = st.session_state.overlay
    elif show_mask_checkbox and show_image_checkbox and not show_overlay_checkbox:
        show_image = st.session_state.batched_mask      
    elif not show_mask_checkbox and show_image_checkbox and show_overlay_checkbox:
        side_tab_options[2].warning(OVERLAY_WARNING)
        show_image = st.session_state.overlay     
    elif show_mask_checkbox and not show_image_checkbox and show_overlay_checkbox:
        side_tab_options[2].warning(OVERLAY_WARNING)
        show_image = st.session_state.overlay    
    elif show_mask_checkbox and show_image_checkbox and show_overlay_checkbox:
        side_tab_options[2].warning(OVERLAY_WARNING)
        show_image = st.session_state.overlay 

    if not show_mask_checkbox and not show_image_checkbox and not show_overlay_checkbox:
        processed_image.empty()
    else:
        if show_overlay_checkbox:
            fig = px.imshow(show_image,height=800,aspect='equal')
            processed_image.plotly_chart(fig,use_container_width=True)
            st.markdown('''
                        <span style="color:#0000FF;font-size:40.5px;font-weight:700"> | Ground Truth | </span> 
                        <span style="color:red;font-size:40.5px;font-weight:700"> | Prediction | </span> 
                        <span style="color:#FF00FF;font-size:40.5px;font-weight:700"> | Overlap | </span> 
                        ''',unsafe_allow_html=True)
        else:
            fig = px.imshow(show_image,height=800,aspect='equal')
            processed_image.plotly_chart(fig,use_container_width=True)
        
    side_tab_options[2].divider()
    # -- [ Get accuracy of the prediction result if sample image is chosen ] -- #
    if ("sample" in st.session_state) or ("gt_mask" in st.session_state):
        if st.session_state.sample:
            if "metrics" in st.session_state:
                df:pd.DataFrame = st.session_state.metrics
                new_tab = "\u2001Metrics\u2001\u2001"
                if new_tab not in st.session_state.menu_tabs:
                    st.session_state.menu_tabs.append(new_tab)
                    st.rerun()
                metrics_viewer(data=df,output=side_tab_options[3])
        else:
            if "metrics" in st.session_state:
                df:pd.DataFrame = st.session_state.metrics
                new_tab = "\u2001Metrics\u2001\u2001"
                if new_tab not in st.session_state.menu_tabs:
                    st.session_state.menu_tabs.append(new_tab)
                    st.rerun()
                metrics_viewer(data=df,output=side_tab_options[3])


def metrics_viewer(data:pd.DataFrame, output):
    """
    NAME: metrics_viewer
    DESC: 
        - Takes in the metrics created when processing images
        - Processes them to make them easier to view
        - stylify the metrics to suite this app
    """
    df = {}
    df["Model"] = st.session_state.model_chosen
    df["Name"] = data[0][0]
    df["Accuracy"] = data[0][1]
    df["Precision"] = data[0][2]
    df["Recall"] = data[0][3]
    df["F1"] = data[0][4]
    df["IoU"] = data[0][5]
    metrics_data = pd.DataFrame.from_dict(data=df.items(),orient="columns")
    metrics_data.columns = [""," "]
    background_1 = ["#81b69d","#06768d"]
    background_2 = ["#92ddc8","#046276"]
    text_col = ["black","#ffffff","#f9f1f1"]
    # -- [ Create the data table ] -- #
    metrics_styled = metrics_data.style.applymap(
        lambda _: f"background-color: {background_1[1]}; color: {text_col[1]};", 
        subset=([0,2,4,6], slice(None))).applymap(
        lambda _: f"background-color: {background_2[1]}; color: {text_col[2]};", 
        subset=([1,3,5], slice(None))
        ).set_properties(
        **{'font-weight':'bold',}).hide(axis = 0).hide(axis = 1)
    with output:
        # -- [ Showcase Data ] -- #
        st.dataframe(metrics_styled, 
            hide_index=True,
            use_container_width=True)
        with st.expander("Metrics Explanation"):
            st.header("Explanation on the metrics shown above")
            # TODO[low]: Finish this by adding explanations on all the metrics used

def mmyolo_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    fig = px.imshow(st.session_state.batched_mask,height=800,aspect='equal',)
    processed_image.plotly_chart(fig,use_container_width=True)

def yolo_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    fig = px.imshow(st.session_state.batched_mask,height=800,aspect='equal',)
    processed_image.plotly_chart(fig,use_container_width=True)



def process_image_pipeline(path_img, image, bar):
    #TODO[low]: Add more options for pipeline (modular for instance and SAM)
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
    # TODO[high]: Add metrics for sample in pipeline 
    st.session_state.detections = detections
    st.session_state.batched_mask = batched_mask

def pipeline_show(processed_image, og_img,sidebar_option_subheader, side_tab_options, main_col_1):
    
    sidebar_option_subheader.subheader("Please choose one of the following options:")
    bounding_box_checkbox = side_tab_options[2].checkbox("Show Bounding Box", value=False)
    show_mask_checkbox = side_tab_options[2].checkbox("Show Mask", value=True)

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
    #processed_image.image(show_image)
    fig = px.imshow(show_image,height=800,aspect='equal')
    processed_image.plotly_chart(fig,use_container_width=True)
    #st.plotly_chart(fig,use_container_width=True)

    #main_col_1.download_button(label="Download", data=download_image(show_image), file_name=st.session_state.uploaded_image.name, mime="image/png")

def main():
    st.sidebar.title("Single Cell Nuclei Segmentation")
    menu_tabs = [
        "\u2001\u2001Settings\u2001\u2001",
        "\u2001Image Upload\u2001\u2001" ,
        "\u2001\u2001Options\u2001\u2001",
    ]
    if "menu_tabs" not in st.session_state:
        st.session_state.menu_tabs = menu_tabs
    side_tabs = st.sidebar.tabs(st.session_state.menu_tabs)       

    # WORKAROUND FOR TEMP ERROR
    temp_path = './temp/'
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
        print("[*]: created temp folder")
    
    # -- [ SETTINGS TAB INFO ] -- #
    side_tabs[0].title("Choose Model:")
    model_option = side_tabs[0].selectbox(label="Choose Model Range",
                                options=('Semantic Segmentation', 'Object Detection', 'Pipeline: Object Detection -> Semantic Segmentation','Custom'),
                                index=None,
                                placeholder="Choose a Model Type",
                                )
    if model_option is not None:
        overall_model = None
        if model_option == "Semantic Segmentation":
            overall_model = side_tabs[0].radio(label=" ",
                                                      options=["U-Net","Deeplabv3+",],
                                                      captions=["Popular in medical research","Newer and and upgrade over U-Net"])
            side_tabs[0].divider()
            side_tabs[0].warning("Semantic segmentation models do not show bounding boxes around the seperate nuclei, it will only show mask on image.")
            side_tabs[0].warning("The model was trained on the MoNuSeg dataset.")
            side_tabs[0].warning("First processing of image will take a bit longer as the model weight will be downloaded.")
        elif model_option == "Object Detection":
            overall_model = side_tabs[0].radio(label=" ",
                                                      options=["MMYOLOv8",
                                                               #"Yolov8",
                                                               ],
                                                      captions=["OpenMMLab implementation of Yolov8",
                                                                #"The original Yolov8"
                                                                ])
            side_tabs[0].divider()
            side_tabs[0].warning("Object detection models do not show mask, only a bounding box around the seperate detected nuceli.")
            side_tabs[0].warning("The model was trained on the MoNuSeg dataset.")
            side_tabs[0].warning("First processing of image will take a bit longer as the model weight will be downloaded.")
        elif model_option == "Pipeline: Object Detection -> Semantic Segmentation":
            overall_model = side_tabs[0].radio(label=" ",
                                                      options=["MMYOLO -> SAM"],
                                                      captions=["Object detection through MMYOLO with Segment anything model (SAM)"])
            side_tabs[0].divider()
            side_tabs[0].warning("The pipeline will use the chosen object detector for the input bounding boxes which will then be used with the segment anything model (SAM) to produce the mask around the detected nuceli.")
            side_tabs[0].warning("The model was trained on the MoNuSeg dataset.")
            side_tabs[0].warning("First processing of image will take a bit longer as the model weight will be downloaded.")
            side_tabs[0].warning("This will take a while to process and show.")
        elif model_option == "Custom":
            st.session_state.custom_model = side_tabs[0].file_uploader("Upload model configuration file (py)", type=["py"])
            st.session_state.custom_weights = side_tabs[0].file_uploader("Upload model weights file (pth)", type=["pth"])
            side_tabs[0].warning("Currently only supports MMSEGMENTATION models")
            overall_model = "custom"
        st.session_state.model_option = overall_model

    # -- [ IMAGE UPLOAD TAB INFO ] -- #
    side_tabs[1].header("Upload nuclei image:")
    if "sample" not in st.session_state:
        st.session_state.sample = False
    # -- [ Disable uploader once upload has been done] -- #
    if 'is_uploaded' not in st.session_state:
        st.session_state.is_uploaded = False
    uploaded_image = side_tabs[1].file_uploader("Upload H&E stained image (png)", type=["png"], disabled=st.session_state.is_uploaded)
    uploaded_gt = side_tabs[1].file_uploader("Upload Ground Truth image (png) (optional)", type=["png"])

    side_tabs[1].divider()
    # -- [ Be able to choose between different datasets which already have images with GT for metrics ] -- #
    side_tabs[1].header("Or, choose from our sample images:")
    side_tabs[1].warning("Sample images will also give metrics for how well the model did on that image [only works for semantic segmentation atm]")
    image_files, images_subset = utils.load_images_test_datasets()
    sets = side_tabs[1].multiselect("Dataset Selector", images_subset, key="dataset_multi")
    view_images = []
    for image_file in image_files:
        if any(set in image_file for set in sets):
            view_images.append(image_file)
        else:
            if "image" in st.session_state and uploaded_image is None:
               del st.session_state.image
    images = []
    for img in view_images:
        with open(img, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
            images.append(f"data:image/png;base64,{encoded}")
    n = 1
    to_show = []
    for i in range(0, int(len(images) / 2.5), n):
        to_show.append(images[i:i+n])
    with side_tabs[1]:
        clicked = clickable_images(to_show,
                                titles=[],
                                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                                img_style={"margin": "5px", "height": "200px"},
                                key="clickable_img"
                                )
        if clicked > -1 and "dataset_multi" in st.session_state:
            if len(view_images) > 0:
                img = cv2.imread(view_images[clicked])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.session_state.image = (img,view_images[clicked])
                st.write(utils.name_processer(str(view_images[clicked])))
                st.session_state.sample = True
    side_tabs[1].warning("Uploaded image takes priority thus if selecting from sample, please remove uploaded file")

    # TODO[medium] - Add multi image input (list of images for processing)
    # -- [ OPTIONS TAB INFO ] -- #
    side_tabs[2].markdown("<h1 style='text-align: center; font-size: 40px'>Options</h1>", unsafe_allow_html=True)
    subheader_text = "Please Process Image for Options"
    sidebar_option_subheader = side_tabs[2].subheader(subheader_text)

    if uploaded_image is not None:
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = uploaded_image
        f = NamedTemporaryFile(dir='./temp', suffix='.png', delete=True)
        f.write(uploaded_image.getbuffer())
        img = cv2.imread(f.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not st.session_state.is_uploaded:
            st.session_state.is_uploaded = True
            st.rerun()
        # -- Define image for session state -- #
        st.session_state.image = (img,f.name)
        if "sample" in st.session_state:
            st.session_state.sample = False
        # -- ------------------------------ -- #
            

############
#  GT MASK 
############
#    gt_path_disk = "./temp/gt/"

    # TODO[HIGH]: Finish upload check for GT adn add to session. ALSO ADD TO PLACES FOR METRICS AND OVERLAY
    if uploaded_gt is not None:
        if "gt_mask" not in st.session_state:     
            # if not os.path.exists(gt_path_disk):
            #     os.mkdir(gt_path_disk)
            #     print("[*] HOME.py | GT path created")
            print("[*] HOME.PY | line: 620 | GT provided")
            f = NamedTemporaryFile(dir="./temp", suffix='.png', delete=False)
            f.write(uploaded_gt.getbuffer())
            st.session_state.gt_mask = f.name
            print(f"Checking if file exists: {os.path.exists(st.session_state.gt_mask)} | {Path(st.session_state.gt_mask).stem}.png")
    else:
        if "gt_mask" in st.session_state:
            if os.path.exists(st.session_state.gt_mask):
                os.remove(st.session_state.gt_mask)
            del st.session_state.gt_mask


    if "image" in st.session_state:
        processed_image = st.empty()
        given_image, img_name = st.session_state.image
        fig = px.imshow(given_image,height=800,)
        processed_image.plotly_chart(fig)
        cols = st.columns(3)
        if cols[2].button('Clear Image', disabled=(uploaded_image is not None)):
            app_rerunner()
            st.rerun()
        # -- Check if button has already been pressed -- #
        if "is_processed" not in st.session_state:
            st.session_state.is_processed = False
        # -- ---------------------------------------- -- #
        if 'process_button' not in st.session_state:
            st.session_state.process_button = False  
        # -- ---------------------------------------- -- #
        if cols[0].button('Process Image', disabled=st.session_state.process_button):
                if "model_option" in st.session_state:
                    if cols[1].button('Stop Processing', disabled=st.session_state.process_button):
                        st.stop()
                    with st.spinner(text='In progress'):
                        bar = st.progress(0)
                        print(img_name)
                        models_selector(st.session_state.model_option)(img_name, given_image, bar)
                        bar.progress(100)
                        st.success('Done')
                        st.session_state.is_processed = True
                else:
                    st.warning("Please Choose a model in the 'Settings' tab on the left (sidebar)")
                    st.stop()
        

        if st.session_state.is_processed:
            st.session_state.process_button = True
            show_selector(st.session_state.model_option)(processed_image=processed_image,
                                                            side_tab_options=side_tabs,
                                                            sidebar_option_subheader=sidebar_option_subheader,
                                                            main_col_1=cols[1],
                                                            og_img=img,
                                                            )
    else:
        app_rerunner()

def app_rerunner():
    # TODO[low]: Recreate rerunner function to chose what not to delete rather than what to delete and this is not very scalable
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
    if "clickable_img" in st.session_state:
        print("[**] deleting clickable image")
        del st.session_state.clickable_img
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

    new_tab = "\u2001Metrics\u2001\u2001"
    if new_tab in st.session_state.menu_tabs:
        st.session_state.menu_tabs.remove(new_tab)

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