import torch
import torch.nn.functional as F
from dataset import UniDataset
from tqdm import tqdm
from globalenv import *
from typing import List
import os

# method defines all!
# if we want to change how vectors are made, set new method, and put it in uni_generate_vectors function
# save path is hard coded
def uni_generate_vectors(
        method: str, # "md", "pca"
        model,
        layers: List[int],
        dataset: UniDataset,
):
    assert dataset.train == True, "Only generate vectors for train dataset"

    # Vectors Save Path
    if not os.path.exists("./Vectors"):
        os.makedirs("./Vectors")
    if not os.path.exists(f"./Vectors/{dataset.task}"):
        os.makedirs(f"./Vectors/{dataset.task}")
    if not os.path.exists(f"./Vectors/{dataset.task}/{method}"):
        os.makedirs(f"./Vectors/{dataset.task}/{method}")

    save_root = f"./Vectors/{dataset.task}/{method}/"

    if method == "md":
        return generate_md_vectors(model, layers, dataset, save_root)
    elif method == "pca":
        return generate_pca_vectors(model, layers, dataset, save_root)
    else:
        raise ValueError(f"Invalid method: {method}")


def generate_md_vectors(
        model,
        layers: List[int],
        dataset: UniDataset,
        save_root: str,
):
    acts = [dict([(l, []) for l in layers]) for _ in range(2)]
    vects = {}
    for input in tqdm(dataset,desc="Get Acts"):
        input = input.to(model.device)
        model.reset_all()
        model.get_logits(input)
        for l in layers:
            act = model.get_activations(l)
            act = act[:,-1,:].detach().cpu()
            acts[0][l].append(act[0])
            acts[1][l].append(act[1])

    torch.save(acts,f"{save_root}acts.pt")
    
    for l in tqdm(layers,desc="Get MD Vectors"):
        acts_0 = torch.stack(acts[0][l])
        acts_1 = torch.stack(acts[1][l])
        vec = (acts_0 - acts_1).mean(dim=0)
        torch.save(vec,f"{save_root}L{l}.pt")
        vects[l] = vec

    return vects


def generate_pca_vectors(
        model,
        layers: List[int],
        dataset: UniDataset,
        save_root: str,
):
    from sklearn.decomposition import  PCA
    ans_num = 2
    acts = [dict([(l, []) for l in layers]) for _ in range(ans_num)]
    vects = {}
    for input in tqdm(dataset,desc="Get Acts"):
        input = input.to(model.device)
        model.reset_all()
        model.get_logits(input)
        for l in layers:
            act = model.get_activations(l)
            act = act[:,-1,:].detach().cpu()
            for i in range(ans_num):
                acts[i][l].append(act[i])
    torch.save(acts,f"{save_root}acts.pt")
        
    for l in tqdm(layers,desc="Get PCA Vectors"):
        all_acts = []
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))

        all_acts_np = torch.cat(all_acts,dim=0).numpy()
        all_acts_np = all_acts_np - all_acts_np.mean(axis=0)
        pca_model = PCA(n_components=1).fit(all_acts_np)
        vec = torch.FloatTensor(pca_model.components_[0])

        direction = all_acts[0].mean(dim=0) - all_acts[-1].mean(dim=0)
        cos_sim = F.cosine_similarity(vec.unsqueeze(0),direction.unsqueeze(0),dim=1).item()
        if cos_sim < 0:
            vec = -vec

        vec *= direction.norm().item()
        torch.save(vec,f"{save_root}L{l}.pt")

        vects[l] = vec
    
    return vects