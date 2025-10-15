from dataset import UniDataset
from typing import List
from utils import *
from tqdm import tqdm
import torch
import numpy as np
import os
from globalenv import *
import json


        
def get_score(
    layers:List[int],
    dataset:UniDataset,
    vec_task:str,
    vec_method:str,
    acts_pre:str="standard",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Whether use GPU to compute score

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

    # Get Activations
    acts = torch.load(f"{vec_root}acts.pt")
    
    # Get Vectors
    score_info = {}
    vects = {}
    vects_norm = {}
    for l in layers:
        vector_path = vec_root + f"L{l}.pt"
        vects[l] = torch.load(vector_path)
        vects_norm[l] = vects[l].norm().item()
        vects[l] /= vects_norm[l]

    for l in tqdm(layers,desc="Get Score"):
        all_acts = []
        all_labels = []
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))
            all_labels.append(torch.ones(all_acts[i].shape[0])*i)

        all_acts = torch.cat(all_acts,dim=0).to(device)

        if acts_pre == "standard":
            all_acts = (all_acts-all_acts.mean(dim=0))/all_acts.std(dim=0)

        all_labels = torch.cat(all_labels,dim=0).to(device)

        # C_score
        all_diff = all_acts[all_labels==0] - all_acts[all_labels==1]
        all_diff = all_diff / all_diff.norm(dim=1, keepdim=True)
        v = vects[l].to(device)
        c_score = all_diff @ v
        c_score = c_score.mean().item()

        # D_score
        n_features = all_acts.size(1)
        mean_total = all_acts.mean(dim=0)
        S_w = torch.zeros((n_features, n_features), device=device)
        S_b = torch.zeros((n_features, n_features), device=device)

        for c in torch.unique(all_labels):
            class_acts = all_acts[all_labels == c]
            mean_class = class_acts.mean(dim=0)
            diff = class_acts - mean_class
            S_w += diff.T @ diff
            mean_diff = (mean_class - mean_total).unsqueeze(1)  # (D,1)
            S_b += class_acts.size(0) * (mean_diff @ mean_diff.T)

        S_t = S_w + S_b
        v = v.reshape(-1, 1)
        numerator = (v.T @ S_b @ v).item()
        denominator = (v.T @ S_t @ v).item()
        d_score = numerator / denominator if denominator != 0 else float('inf')
        
        score_info[l] = {"s_score":d_score+c_score,"d_score":d_score,"c_score":c_score}
        with open(f"{save_root}L{l}.json","w") as f:
            json.dump(score_info[l],f,indent=4)

    return score_info
