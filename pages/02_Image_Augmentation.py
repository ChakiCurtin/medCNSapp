import streamlit as st
from tempfile import NamedTemporaryFile
import cv2
from streamlit_image_comparison import image_comparison
import torchstain
from torchvision import transforms
import numpy as np
st.set_page_config(page_title="Image Augmentation", 
                   initial_sidebar_state="expanded", 
                   layout="wide")

# -- ## -- CUSTOM CSS -- ## -- #
# -- [ "Remove the "made with streamlit" at the footer of the page]
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def preproc_selector(preproc_option: str):
    dataset_dict = {
        "stain normalisation": stain_norm,
        "clahe": clahe,
    }
    return dataset_dict.get(preproc_option)

def stain_norm(fit_image, given_img):
    # https://github.com/EIDOSLAB/torchstain
    target = cv2.cvtColor(fit_image, cv2.COLOR_BGR2RGB)
    to_transform = cv2.cvtColor(given_img, cv2.COLOR_BGR2RGB)

    T = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*255)])
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))
    t_to_transform = T(to_transform)
    norm, _,_ = normalizer.normalize(I=t_to_transform)
    norm = norm.numpy().astype(np.uint8)
    #norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR) # this is supposed to work but it makes the image go back to RGB? Works without it for now
    return norm


def clahe(image, clahe_thresh):
    # -- [ Set up CLAHE ] -- #
    clahe = cv2.createCLAHE(clipLimit=clahe_thresh, tileGridSize=(8,8)) # default tilegrid
    assert image is not None, "Image File not recognised, check with os.path.exists()"
    # -- [ Apparently, CLAHE cannot be done to RGB directly so convert to LAB and apply then revert back ] -- #
    # -- [ https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images ] -- #
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def main():
    # TODO: ADD SAMPLE IMAGES HERE 
    st.markdown(
    """
    <style>
    div.stMarkdown {
        text-align:center;
        }
  </style>
    """, unsafe_allow_html=True)
    cols = st.columns(5)
    # TODO [medium]: Create stain norm technique
    # Allow one image to selected as the fit, allow for uploaded image 
    # and show comparison from original and stain norm using slider
    preproc_option = st.sidebar.radio(
        "Select Augmentation Technique",
        ["CLAHE", "Stain Normalisation"],
        captions=None,
    )
    if preproc_option == "CLAHE":
        st.markdown("**Contrast Limited Adaptive Histogram Equalization (CLAHE)**")
    elif preproc_option == "Stain Normalisation":
        st.markdown("**Stain Normalisation**")
        uploaded_fit = st.sidebar.file_uploader("Upload FIT H&E stained image (png)", type=["png"])
        st.sidebar.warning("Both Fit image and image to compare needs to be given")
    st.sidebar.divider()
    if 'is_uploaded' not in st.session_state:
        st.session_state.is_uploaded = False
    uploaded_image = st.sidebar.file_uploader("Upload H&E stained image (png)", type=["png"], disabled=st.session_state.is_uploaded)
    st.sidebar.divider()
    st.sidebar.write("To load a new image, please press the 'x' to remove current image ^")
    st.sidebar.divider()
    if uploaded_image is not None:
        if 'uploaded_image' not in st.session_state:
             st.session_state.uploaded_image = uploaded_image
        with NamedTemporaryFile(dir='./temp', suffix='.png') as f:
            f.write(uploaded_image.getbuffer())
            img = cv2.imread(f.name)
            if not st.session_state.is_uploaded:
                st.session_state.is_uploaded = True
                st.rerun()
            
            # -- Define image for session state -- #
            if 'pre_image' not in st.session_state:
                st.session_state.pre_image = img
            # -- ------------------------------ -- #
            #processed_image.image(st.session_state.pre_image, caption=uploaded_image.name, width=600,)
            if preproc_option == "CLAHE":
                # -- [ RUN THE CLAHE ALG ] -- #
                thresh = st.slider("Choose CLAHE Threshold", 0.01, 10.0, 1.0,)
                with cols[1]:
                    clahed_image = clahe(st.session_state.pre_image.copy(), thresh)
                    image_comparison(
                        img1=st.session_state.pre_image,
                        img2=clahed_image,
                        label1="Original",
                        label2="CLAHE: " + str(thresh),
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True,
                        width=700,
                    )
            elif preproc_option == "Stain Normalisation":
                if uploaded_fit is not None:
                    print("[*] Both FIT and showing image given...Stain Norm")
                    with NamedTemporaryFile(dir='./temp', suffix='.png') as g:
                        g.write(uploaded_fit.getbuffer())
                        fit_img = cv2.imread(g.name)
                        original_img = cv2.cvtColor(st.session_state.pre_image, cv2.COLOR_RGB2BGR)
                        with cols[1]:
                            result_img = stain_norm(fit_img,st.session_state.pre_image.copy())
                            image_comparison(
                                img1=original_img,
                                img2=result_img,
                                label1="Original",
                                label2="Stain Normalised",
                                starting_position=50,
                                show_labels=True,
                                make_responsive=True,
                                in_memory=True,
                                width=700,
                            )



    else:
        if st.session_state.is_uploaded:
            st.session_state.clear()
            st.rerun()
        print("[*] Image_AUG: clearing var")




if __name__ == "__main__":
    main()