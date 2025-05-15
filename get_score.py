import sys
from dataset import UniDataset
from typing import List, Dict
from utils import *
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from globalenv import *
import json
from get_results import get_raw_results_file_path


        
def get_score(
    layers:List[int],
    dataset:UniDataset,
    vec_task:str,
    vec_method:str,
    acts_pre:str=None,
):
    assert dataset.train == True, "Only Use Train Dataset"

    vec_root = f"./Vectors/{vec_task}/{vec_method}/"
    
    svec_path = vec_root[10:] # remove "./Vectors/"
    svec_path = svec_path.replace("/","+")
    

    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    # Score Save Path
    if not os.path.exists(f"./Score{acts_pre_str}"):
        os.mkdir(f"./Score{acts_pre_str}")
    if not os.path.exists(f"./Score{acts_pre_str}/{dataset.task}"):
        os.mkdir(f"./Score{acts_pre_str}/{dataset.task}")
    if not os.path.exists(f"./Score{acts_pre_str}/{dataset.task}/{svec_path}"):
        os.mkdir(f"./Score{acts_pre_str}/{dataset.task}/{svec_path}")

    save_root = f"./Score{acts_pre_str}/{dataset.task}/{svec_path}/"


    ans_num = 2

    # Get Acts
    acts = torch.load(f"{vec_root}acts.pt")
    
    # Get Vectos
    proj_info = {}
    vects = {}
    vects_norm = {}
    for l in layers:
        vector_path = vec_root + f"L{l}.pt"
        vects[l] = torch.load(vector_path)
        vects_norm[l] = vects[l].norm().item()
        vects[l] /= vects_norm[l]

    for l in tqdm(layers,desc="Get Proj"):
        all_acts = []
        all_labels = []
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))
            all_labels.append(torch.ones(all_acts[i].shape[0])*i)

        all_acts = torch.cat(all_acts,dim=0).numpy()

        if acts_pre == "standard":
            all_acts = (all_acts-all_acts.mean(axis=0))/all_acts.std(axis=0)

        all_labels = torch.cat(all_labels,dim=0).numpy()

        all_diff = all_acts[all_labels==0] - all_acts[all_labels==1]
        all_diff = all_diff / np.linalg.norm(all_diff,axis=1,keepdims=True)
        c_score = float(np.mean(all_diff.dot(vects[l].numpy())))

        n_features = all_acts.shape[1]
        mean_total = all_acts.mean(axis=0)
        S_w = np.zeros((n_features, n_features), dtype=all_acts.dtype)
        S_b = np.zeros((n_features, n_features), dtype=all_acts.dtype)
        classes = np.unique(all_labels)
        for c in classes:
            class_acts = all_acts[all_labels == c]
            mean_class = class_acts.mean(axis=0)
            diff = class_acts - mean_class
            S_w += diff.T @ diff
            mean_diff = (mean_class - mean_total).reshape(-1, 1)
            S_b += class_acts.shape[0] * (mean_diff @ mean_diff.T)
        S_t = S_w + S_b    
        v = vects[l].numpy().reshape(-1, 1)
        numerator = (v.T @ S_b @ v).item()
        denominator = (v.T @ S_t @ v).item()
        d_score = numerator / denominator if denominator!= 0 else np.inf
        
        proj_info[l] = {"s_score":d_score+c_score,"d_score":d_score,"c_score":c_score}
        with open(f"{save_root}L{l}.json","w") as f:
            json.dump(proj_info[l],f,indent=4)

    return proj_info