from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import numpy as np
from globalenv import *
import random

anth_all_path = "./Dataset/Anth_Persona_ALL/{}_{}.jsonl"

class UniDataset(Dataset):
    def __init__(self,
                 task:str, 
                 train:bool,# use for generate vectors or test
                 set:str, # "train", "val", "test"
                 batch_size:int=50,): 

        self.task = task
        self.train = train
        self.set = set
        self.batch_size = batch_size


        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.keys = [] # List[1|-1], 1: Ans[1] match the task behavior; -1: Ans[-1] match
        self.data = self.__load_data()
        # Train length = question nums, each batch has same question with different ans
        # Test length = 1, only one batch with all questions

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __tok(self, input):
        return self.tokenizer(input,padding="longest",return_tensors="pt")
    
    def __load_data(self):
        return self.__load_anth_data()
    
    
    def __load_anth_data(self):
        data_toks = []
        data_path = anth_all_path.format(self.task,self.set)

        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
        Ans = [" Yes"," No"]
        
        data_len = len(data)

        # Get Tokens
        if self.train:
            # Default Batch_size=1, hard coded
            for i in tqdm(range(data_len),desc="Get Tokens for Train"):
                cur_data = []
                cur_Ans = Ans if data[i]["answer_matching_behavior"] == " Yes" else Ans[::-1]
                cur_key = 1 if data[i]["answer_matching_behavior"] == " Yes" else -1
                self.keys.append(cur_key)

                for ans in cur_Ans:
                    cur_data.append(INST_TEMPLATE.format(data[i]["question"]+Anth_YN_QUESTION_PROFIX,INST_SYS_ANS_PREFIX_alt1+ans))
                cur_data = self.__tok(cur_data)
                data_toks.append(cur_data)

        else: # Test, No Ans
            for i in tqdm(range(0,data_len,self.batch_size),desc="Get Tokens for Test"):
                cur_data = []
                for j in range(self.batch_size):
                    cur_idx = i+j
                    if cur_idx >= data_len:
                        break
                    cur_key = 1 if data[cur_idx]["answer_matching_behavior"] == " Yes" else -1
                    self.keys.append(cur_key)

                    if self.prompt_mode == "inst":
                        cur_data.append(INST_TEMPLATE.format(data[cur_idx]["question"]+Anth_YN_QUESTION_PROFIX,INST_SYS_ANS_PREFIX_alt1))
                    else:
                        raise ValueError(f"Invalid prompt mode: {self.prompt_mode}")
                cur_data = self.__tok(cur_data)
                data_toks.append(cur_data)
        return data_toks
             
def get_keys(task:str,set:str):
    keys = []
    data_path = anth_all_path.format(task,set)
    
    with open(data_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    data_len = len(data)
    for i in range(data_len):
        keys.append(1 if data[i]["answer_matching_behavior"] == " Yes" else -1)
    return keys
