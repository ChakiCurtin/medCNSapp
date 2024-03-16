import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils import load_images
# SOME LINKS FOR REFERENCE:
# * https://plotly.com/python-api-reference/generated/plotly.express.pie
# * https://plotly.com/python/builtin-colorscales/
# * https://github.com/Delgan/loguru
# * 
st.set_page_config(page_title="Dataset Selector", 
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

def dataset_selector(dataset_option: str):
    dataset_dict = {
        "monuseg": monuseg,
        "monusac": monusac,
        "cryonuseg": cryonuseg,
        "tnbc": tnbc,
    }
    return dataset_dict.get(dataset_option)

@st.cache_data
def load_dataset(filename:str):
    mod_path = Path(__file__).parents[1]
    file = open(f"{mod_path}/dataset_util_files/{filename}", "r")
    dataset_dict = {
        "Patient ID": [],
        "Organ": [],
        "Set": [],
        "Num of Organ": [],
        "Disease type": [],
    }
    for line in file:
        line_arr = line.split(",")
        patient_id = line_arr[0]
        data_set = line_arr[1]
        organ = line_arr[2]
        disease = line_arr[3]
        dataset_dict["Patient ID"].append(patient_id)
        dataset_dict["Organ"].append(organ)
        dataset_dict["Set"].append(data_set)
        dataset_dict["Num of Organ"].append(1)
        dataset_dict["Disease type"].append(disease)
    file.close()
    df = pd.DataFrame.from_dict(dataset_dict)
    return df


def monusac():
    df = load_dataset("monusac.txt")
    df_mutate = df.copy(deep=True)
    df_mutate.drop(columns='Num of Organ', axis=1, inplace=True)
    cols = st.columns(2)
    url = "https://monusac-2020.grand-challenge.org/Data/"
    cols[0].write("""About MoNuSAC:
- Publicly available Cell Nuceli Segmentation (CNS) dataset containing H&E stained WSIs
- Contains over 46000 annotations covering 71 patients in a total of 31 hospitals
- covers 4 organs, with 46 training WSI (split ROI) and 25 testing WSI (ROI taken)
- distinguishes cells from 4 nuclei types (Epithelial, Lymphocytes, Macrophages and Neutrophils)
- Images do not have a set size, some are smaller and some are wider
- All Images come from The Cancer Genome Atlas, which is the world's largest digital slide repository
- Originally created for The Grand Challenge, it is a well known dataset for nuclei segmentation
- Full MoNuSAC dataset download [link](%s).""" % url)
    cols[0].divider()
    #cols[0].checkbox("Use container width", value=True, key="use_container_width")
    cols[0].dataframe(df_mutate, use_container_width=True, hide_index=True,)
    #creating pie chart for number of organs
    pie_chart = px.pie(df, 
                       title="Total number of organs covered",
                       values="Num of Organ",
                       names="Organ",
                       hole=0.4,
                       width=700,
                       height=550,
                       color_discrete_sequence=px.colors.sequential.Turbo,
                       #color_discrete_sequence=px.colors.sequential.RdBu,
                       )
    cols[1].plotly_chart(pie_chart)
    cols[1].divider()
    # -- [ Show tiles of images from MoNuSeg ] -- #

    # image_files, images_subset = load_images()
    # sets = cols[1].multiselect("Dataset Image Select set(s)", images_subset)
    # view_images = []
    # for image_file in image_files:
    #     if any(set in image_file for set in sets):
    #         view_images.append(image_file)
    
    # n = 3
    # groups = []
    # for i in range(0, len(view_images), n):
    #     groups.append(view_images[i:i+n])

    # for group in groups:
    #     colss = cols[1].columns(n)
    #     for i, image_file in enumerate(group):
    #         colss[i].image(image_file)


def monuseg():
    df = load_dataset("monuseg.txt")
    df_mutate = load_dataset("monuseg.txt")
    df_mutate.drop(columns='Num of Organ', axis=1, inplace=True)
    cols = st.columns(2)
    url = "https://monuseg.grand-challenge.org/Data/"
    # a quick about section for MoNuSeg:
    cols[0].write("About MoNuSeg: \n\nMoNuSeg is a publicly available Cell Nuceli Segmentation (CNS) dataset containing H&E stained WSIs." +
                  " It contains 51 images, with 37 in the training set and 14 in the test set." +
                  " It provides 30837 annotations, with 24140 in the training set" +
                  " and 6697 in the test set. This is a binary segmentation dataset, so" +
                  " nuclei are all considered to be the same. Every image comes" +
                  " as a 1000 × 1000 pixel region of interest (ROI)." +
                  " All Images come from The Cancer Genome Atlas, which is the world's largest digital slide repository." + 
                  " Originally created for The Grand Challenge, it is a well known dataset for nuclei segmentation." +
                  " Full MoNuSeg dataset download [link](%s)." % url 
                  )
    cols[0].divider()
    cols[0].dataframe(df_mutate, use_container_width=True, hide_index=True,)
    
    #creating pie chart for number of organs
    pie_chart = px.pie(df, 
                       title="Total number of organs covered",
                       values="Num of Organ",
                       names="Organ",
                       hole=0.4,
                       width=700,
                       height=550,
                       color_discrete_sequence=px.colors.sequential.Turbo,
                       #color_discrete_sequence=px.colors.sequential.RdBu,
                       )
    cols[1].plotly_chart(pie_chart)
    cols[1].divider()

    # # -- [ Show tiles of images from MoNuSeg ] -- #
    image_files, images_subset = load_images()
    sets = cols[1].multiselect("Dataset Image Select set(s)", images_subset)
    view_images = []
    for image_file in image_files:
        if any(set in image_file for set in sets):
            view_images.append(image_file)
    
    n = 3
    groups = []
    for i in range(0, len(view_images), n):
        groups.append(view_images[i:i+n])

    for group in groups:
        colss = cols[1].columns(n)
        for i, image_file in enumerate(group):
            colss[i].image(image_file)

#TODO[medium]: Finish the rest of the datasets. Find and add information for each
def cryonuseg():
    cols = st.columns(2)
    url = "https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images?select=tissue+images"
    cols[0].write("""About CryoNuSeg:
                - publicly available CNS dataset
                - The first frozen tissue sample dataset to be fully annotated
                - It contains 30 whole slide images from 10 organs.
                - Since it is frozen section tissue samples, the cell quality can suffer. This one is a high-quality FS dataset.
                - nuclei are all considered to be the same.
                - as a 1000 × 1000 pixel region of interest (ROI).
                - Full CryoNuSeg dataset download [link](%s).""" % url)
    cols[0].divider()
def tnbc():
    st.write("TNBC WIP")

def main():
    dataset_option = st.sidebar.selectbox(
        label="Choose Dataset to view:",
        options=('MoNuSeg', 
                 'MoNuSAC', 
                 #'CryoNuSeg', 
                 #'TNBC',
                 ),
        index=0,
        placeholder="Choose a Dataset",
        )
    cols = st.columns(3)
    cols[1].header(f"{dataset_option} Dataset")
    dataset_selector(dataset_option=str(dataset_option).lower())() # makes a function from dict of functions
    st.sidebar.divider()


if __name__ == "__main__":
    main()