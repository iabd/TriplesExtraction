import os

class Config:
    pretrained = True
    num_labels = 4
    model = "bert-base-cased"
    #checkpoints = [f"2021-07-24/{i}" for i in os.listdir("2021-07-24") if "NER_bert-base-cased_ fold -" in i]
