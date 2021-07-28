import argparse, spacy
from model import Transformer
import pandas as pd
from config import Config
from dataset import get_and_process_data, easy_extraction, TestDataset
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from utils import *


nlp=spacy.load('en')

def load(model, with_checkpoint=None):
    model=Transformer(model)
    if with_checkpoint:
        checkpoint=torch.load(with_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded!", end="\r")
    return model


def extract_news20_dep():
    """Main function for extracting triples of News20 datasets using dependency tree."""
    try:
        df = pd.read_csv("data/20NewsGroups.csv")
    except:
        print(
            "Data csv not found. Please make sure its data/20NewsGroups.csv. \nLoading from sklearn. This could take a while...")
        df = get_and_process_data()

    all_triples = []
    for idx, sentences in enumerate(df['processed']):
        print(f"Extracting triples .. {idx}/{len(df)}", end="\r")
        try:
            sentences = eval(sentences)
        except:
            pass

        triples = []
        for sentence in sentences:
            triple = easy_extraction(nlp(sentence), modify_relations=True)
            if triple:
                if " " not in triple[0]:
                    triples.append(triple)

        all_triples.append(triples)

    df['triples'] = all_triples
    df.to_csv("data/20NewsGroups.csv", index=False)
    print("Done!!! The extracted triples are in 20NewsGroups.csv file.")



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', default='dep', choices=['dep', 'bert'], help="Extraction choice! Currently supports 'dep' for using dependency tree and 'bert' for using transformer model. ")
    args = parser.parse_args()

    if args.mode=="dep":
        extract_news20_dep()

    if args.mode=="bert":
        try:
            df = pd.read_csv("data/20NewsGroups.csv")
        except:
            print(
                "Data csv not found. Please make sure its data/20NewsGroups.csv. \nLoading from sklearn. This could take a while...")
            df = get_and_process_data()

        df["num_sentences"] = df['processed'].map(eval).map(len)
        tokenizer = BertWordPieceTokenizer(
                    "HuggingFace/Bert/bert_base_uncased_vocab.txt",
                    lowercase=False
                )
        tokens = {
                'cls': tokenizer.token_to_id('[CLS]'),
                'sep': tokenizer.token_to_id('[SEP]'),
                'pad': tokenizer.token_to_id('[PAD]'),
            }
        print(">> Loading model..")
        model=load(Config.model, with_checkpoint=Config.checkpoints[0])
        news20 = []
        dset = TestDataset(df, tokenizer, tokens)
        dloader = DataLoader(dset, batch_size=1)

        for idx, batch in enumerate(dloader):
            print(f"Batch : {idx}/{len(dloader)}", end="\r")
            try:
                output = model(batch['input_ids'])
                news20.append(decode_output(output, batch['offsets'], batch['data']))
            except:
                news20.append([])
                pass

        news20_triples = []
        rows = []
        for idx, triple in enumerate(news20):
            if "" in triple or " " in triple:
                continue
            else:
                news20_triples.append(triple)
                rows.append(dset.row_number[idx])

        df1 = pd.DataFrame()
        df1['triples'] = news20_triples
        df1['row'] = rows
        df1 = df1.groupby('row')['triples'].apply(list).reset_index(drop=True)
        df['bert_triples']=df1['triples']
        df.to_csv("data/20NewsGroups.csv", index=False)
        print("Done!!! The extracted triples are in 20NewsGroups.csv file.")