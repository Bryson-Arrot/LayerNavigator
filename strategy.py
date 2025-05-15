import json
from globalenv import *
import random


class UniStrategy:
    def __init__(
            self,
            task:str,
            strategy:str,
            num_layers:int,
            method:str,
    ):
        self.task = task
        self.strategy = strategy
        self.num_layers = num_layers

        self.layers = []
        self.scores = []
        self.__get_layers()
    
    def __get_layers(self):

        if self.strategy == "my":
            s_scores = {}
            for i in LAYERS:
                with open(f"./Score-standard/{self.task}/{self.task}+{self.method}/L{i}.json","r") as f:
                    s_scores[i] = json.load(f)["s_score"]
            sorted_layers = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
            self.layers = [i[0] for i in sorted_layers]
            self.layers = self.layers[:self.num_layers]
            self.scores = [i[1] for i in sorted_layers]
            self.scores = self.scores[:self.num_layers]
        
        elif self.strategy.startswith("my-p"):
            p = float(self.strategy.split("-p")[-1])
            assert 0<p<1
            s_scores = {}
            for i in LAYERS:
                with open(f"./Score-standard/{self.task}/{self.task}+{self.method}/L{i}.json","r","r") as f:
                    info = json.load(f)
                    s_scores[i] = info["d_score"]*p + info["c_score"]*(1-p)
            sorted_layers = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
            self.layers = [i[0] for i in sorted_layers]
            self.layers = self.layers[:self.num_layers]
            self.scores = [i[1] for i in sorted_layers]
            self.scores = self.scores[:self.num_layers]     
          
        

                    
