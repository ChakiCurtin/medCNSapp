import os
from pathlib import Path
import argparse
import requests
import json

# https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/
# https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/
# https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/#introducing-search-and-retrieval-requests


def search_tcga(patient_id):
    """
    INPUT: patient_id (str)
    OUTPUT: (organ_name, disease_type)
    """
    case_endpt = "https://api.gdc.cancer.gov/cases/"

    fields = [
        #"submitter_id",
        #"project.project_id",
        #"samples.submitter_id",
        #"submitter_slide_ids",
        "primary_site",
        "tissue_source_site.project",
        #"project.name",
        #"disease_type",
        #"diagnoses.vital_status"
    ]
    fields = ','.join(fields)
    filters = {
        "op": "in",
        "content":{
            "field": "submitter_slide_ids",
            "value": [patient_id]
            }
        }

    params = {  
        "size":"10",
        "pretty":"TRUE",
        "fields":fields,
        "format":"TSV",
        "filters": json.dumps(filters),
    }
        # TODO[low/medium]: find way to check if id is not in TCGA
    response = requests.get(case_endpt, params=params)
    target = response.content.decode("utf-8")
    #print(target)
    #print(patient_id)
    total = target.split("\t")
    organ = total[3]
    diseasevar = total[4]
    diseasevar = diseasevar.split("\r\n")
    disease = diseasevar[0]
    return (organ,disease)


def main(args):
    image_path_root = args.dataset_path
    sett = ["train", "val", "test"]
    checker = "*.png"
    
    image_path_train = image_path_root / sett[0]
    image_path_val = image_path_root / sett[1]
    image_path_test = image_path_root / sett[2]

    image_train = sorted(image_path_train.glob(checker))
    image_val = sorted(image_path_val.glob(checker))
    image_test = sorted(image_path_test.glob(checker))

    images = [image_train,image_val,image_test]

    print(f"[*] Number of images in train set: {len(image_train)}, val set: {len(image_val)}, test set: {len(image_test)}")

    # check for train, val, test nones
    name_dict = {}
    patient_dict = {}
    # get all
    for x in range(3):
        if len(images[x]) > 0:
            for name_path in images[x]:
                name_splits = name_path.stem
                name_splits = name_splits.split("_")
                name = name_splits[0]
                id_split = name.split("-")
                name = f"{id_split[0]}-{id_split[1]}-{id_split[2]}-{id_split[3]}-{id_split[4]}-{id_split[5]}"
                
                if name not in name_dict:
                    organ,disease = search_tcga(name)
                    name_dict[name] = (organ,disease)
                    # add to patients dict touple (set,primary_organ,disease)
                    patient_dict[name_path.stem] = (sett[x], organ, disease)
                else:
                    organ,disease = name_dict.get(name)
                    patient_dict[name_path.stem] = (sett[x], organ, disease)
                print(f"[*] Patient: {name_path.stem} | Set: {sett[x]} | Organ: {organ} | disease: {disease}")

    # After API call checking. Length of patient dict = train + val + test of dataset
    print(f"[*] Patient dictionary size: {len(patient_dict)}")
    length_dataset = len(image_train) + len(image_val) + len(image_test)
    if len(patient_dict) == length_dataset:
        # Now save these into a file
        f = open(args.save, "w")
        for key in patient_dict:
            data_set,organ,disease = patient_dict[key]
            to_string = f"{key},{data_set},{organ},{disease}\n"
            f.write(to_string)
        f.close()
    else:
        print(f"[!] Something went wrong....dict: {len(patient_dict)} != {length_dataset} in dataset")
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=None, required=True, help="Path to dataset folder.")
    parser.add_argument("--save", type=Path, default=None, required=True, help="The path + name to where output file should be created")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)