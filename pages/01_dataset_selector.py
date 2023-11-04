import streamlit as st
import pandas as pd
import plotly.express as px
import glob

# SOME LINKS FOR REFERENCE:
# * https://plotly.com/python-api-reference/generated/plotly.express.pie
# * https://plotly.com/python/builtin-colorscales/
# * https://github.com/Delgan/loguru
# * 

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

def dataset_selector(dataset_option: str):
    dataset_dict = {
        "monuseg": monuseg,
        "monusac": monusac,
        "cryonuseg": cryonuseg,
        "tnbc": tnbc,
    }
    return dataset_dict.get(dataset_option)

@st.cache_data
def load_monuseg():
    return pd.DataFrame.from_dict(
        {
        "Patient ID": ["TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1", "TCGA-AR-A1AK-01Z-00-DX1", "TCGA-AR-A1AS-01Z-00-DX1",
                       "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1", "TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1",
                       "TCGA-HE-7129-01Z-00-DX1", "TCGA-HE-7130-01Z-00-DX1","TCGA-B0-5710-01Z-00-DX1","TCGA-B0-5698-01Z-00-DX1",
                       "TCGA-18-5592-01Z-00-DX1","TCGA-38-6178-01Z-00-DX1","TCGA-49-4488-01Z-00-DX1","TCGA-50-5931-01Z-00-DX1",
                       "TCGA-21-5784-01Z-00-DX1","TCGA-21-5786-01Z-00-DX1","TCGA-G9-6336-01Z-00-DX1","TCGA-G9-6348-01Z-00-DX1",
                       ],
        "Organ": ["Breast", "Breast", "Breast", "Breast", "Breast", 
                  "Breast","Kidney", "Kidney", "Kidney", "Kidney",
                  "Kidney","Kidney","Liver","Liver","Liver","Liver","Liver",
                  "Liver","Prostate","Prostate",
                  ],
        "Num of Organ": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
        "Disease type": ["Breast invasive carcinoma","Breast invasive carcinoma","Breast invasive carcinoma",
                         "Breast invasive carcinoma","Breast invasive carcinoma","Breast invasive carcinoma",
                         "Kidney renal clear cell carcinoma","Kidney renal papillary cell carcinoma",
                         "Kidney renal papillary cell carcinoma","Kidney renal papillary cell carcinoma",
                         "Kidney renal clear cell carcinoma","Kidney renal clear cell carcinoma",
                         "Lung squamous cell carcinoma","Lung adenocarcinoma","Lung adenocarcinoma",
                         "Lung adenocarcinoma","Lung squamous cell carcinoma","Lung squamous cell carcinoma",
                         "Prostate adenocarcinoma","Prostate adenocarcinoma",
                         ],
        }, orient="columns")

def monuseg():
    df = load_monuseg()
    df_mutate = load_monuseg()
    df_mutate.drop(columns='Num of Organ', axis=1, inplace=True)
    cols = st.columns(2)
    # a quick about section for MoNuSeg:
    cols[0].write("About MoNuSeg: \n\nMoNuSeg is a publicly available Cell Nuceli Segmentation (CNS) dataset containing H&E stained WSIs." +
                  " It contains 51 images, with 37 in the training set and 14 in the test set." +
                  " It provides 30837 annotations, with 24140 in the training set" +
                  " and 6697 in the test set. This is a binary segmentation dataset, so" +
                  " nuclei are all considered to be the same. Every image comes" +
                  " as a 1000 × 1000 pixel region of interest (ROI)." +
                  " All Images come from The Cancer Genome Atlas, which is the world's largest digital slide repository." + 
                  " Originally created for The Grand Challenge, it is a well known dataset for nuclei segmentation."
                  )
    cols[0].divider()
    #cols[0].checkbox("Use container width", value=True, key="use_container_width")
    cols[0].dataframe(df_mutate, use_container_width=True, hide_index=True,)
    
    #creating pie chart for number of organs
    pie_chart = px.pie(df, 
                       title="Total number of organs covered",
                       values="Num of Organ",
                       names="Organ",
                       hole=0.4,
                       color_discrete_sequence=px.colors.sequential.Turbo,
                       )
    cols[1].plotly_chart(pie_chart)
    cols[1].divider()
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
        
    


def cryonuseg():
    st.write("CryoNuSeg WIP")

def monusac():
    st.write("MoNuSAC WIP")

def tnbc():
    st.write("TNBC WIP")

def main():
    st.set_page_config(page_title="Dataset Selector", initial_sidebar_state="expanded", layout="wide")
    dataset_option = st.sidebar.selectbox(
        label="Choose Dataset to view:",
        options=('MoNuSeg', 'MoNuSAC', 'CryoNuSeg', 'TNBC'),
        index=0,
        placeholder="Choose a Dataset",
        )
    cols = st.columns(3)
    cols[1].header(f"{dataset_option} Dataset")
    dataset_selector(dataset_option=str(dataset_option).lower())() # makes a function from dict of functions


if __name__ == "__main__":
    main()