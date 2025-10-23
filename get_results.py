import torch
from typing import List
from dataset import UniDataset
import os
from utils import *
import json
from globalenv import *


# All functions are about *test* dataset

def get_raw_BASE_results(
    model,
    test_dataset:UniDataset,
    max_new_tokens:int=200,
):
    assert test_dataset.train == False, "Only Use Test Mode"
    assert test_dataset.set in ["test","val"], "Only Use Test Dataset"


    # Results Save Path
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}")


    save_root = f"./Results/{test_dataset.task}-{test_dataset.set}/"
    save_path = f"{save_root}base.json"

    # Get Text Generation
    model.reset_all()
    results = []
    for d in test_dataset:
        cur_results = model.generate(d.to(model.device),max_new_tokens=max_new_tokens) # raw results
        results.extend(cur_results)
    with open(save_path,"w") as f:
        json.dump(results,f,indent=4)

    # Get Logits
    last_logits = []
    for d in test_dataset:
        cur_last_logits = model.get_last_logits(d.to(model.device)).detach().cpu()
        last_logits.extend(cur_last_logits)
    last_logits = torch.stack(last_logits,dim=0)
    last_logits = torch.softmax(last_logits,dim=-1)
    last_logits = last_logits[:,[QUICK_TOKEN_ID_DICT[" Yes"],QUICK_TOKEN_ID_DICT[" No"]]]
    keys = test_dataset.keys
    total = 0
    assert len(keys) == len(last_logits)
    for i in range(len(keys)):
        if keys[i] == 1:
            total += last_logits[i,0].item()/(last_logits[i,0].item()+last_logits[i,1].item())
        else:
            total += last_logits[i,1].item()/(last_logits[i,0].item()+last_logits[i,1].item())
    print(f"Prob: {total/len(keys)}")

    return total/len(keys)

def get_raw_results(
    model,
    layers:List[int],
    test_dataset:UniDataset,
    Alphas:List[float],
    train_task : str,
    train_method:str,
    max_new_tokens:int=200,
):
    assert test_dataset.train == False, "Only Use Test Dataset"
    assert test_dataset.set in ["val","test"], "Only Use Val and Test Dataset"
    assert len(layers) == len(Alphas), "layers, Alphas must have same length"


    vec_root = f"./Vectors/{train_task}/{train_method}"

    svec_path = vec_root[10:] # remove "./Vectors/"
    svec_path = svec_path.replace("/","+")

    vects = {}
    for i in range(len(layers)):
        vector_path = vec_root + f"/L{layers[i]}.pt"
        vects[layers[i]] = torch.load(vector_path).to(model.device)
        vects[layers[i]] *= Alphas[i]


    # Results Save Path
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}")
    if not os.path.exists(f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}"):
        os.mkdir(f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}")
    

    save_root = f"./Results/{test_dataset.task}-{test_dataset.set}/{svec_path}/"
    file_name = "Res"
    for i in range(len(layers)):
        file_name += f"_L{layers[i]}x{Alphas[i]*100:.0f}"
    save_path = f"{save_root}{file_name}.json"


    # Get Results
    model.reset_all()
    for l in layers:
        model.set_add_activations(l,vects[l])

    results = []
    for d in test_dataset:
        cur_results = model.generate(d.to(model.device),max_new_tokens=max_new_tokens) # raw results
        results.extend(cur_results)

    with open(save_path,"w") as f:
        json.dump(results,f,indent=4)
    
    # Get Logits
    last_logits = []
    for d in test_dataset:
        cur_last_logits = model.get_last_logits(d.to(model.device)).detach().cpu()
        last_logits.extend(cur_last_logits)
    last_logits = torch.stack(last_logits,dim=0)
    last_logits = torch.softmax(last_logits,dim=-1)
    last_logits = last_logits[:,[QUICK_TOKEN_ID_DICT[" Yes"],QUICK_TOKEN_ID_DICT[" No"]]]
    keys = test_dataset.keys
    total = 0
    assert len(keys) == len(last_logits)
    for i in range(len(keys)):
        if keys[i] == 1:
            total += last_logits[i,0].item()/(last_logits[i,0].item()+last_logits[i,1].item())
        else:
            total += last_logits[i,1].item()/(last_logits[i,0].item()+last_logits[i,1].item())
    print(f"Prob: {total/len(keys)}")

    return total/len(keys)