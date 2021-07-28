import torch, re, spacy, neuralcoref
from torch.utils.data import Dataset
from nltk import sent_tokenize
nlp=spacy.load('en')
neuralcoref.add_to_pipe(nlp,greedyness=0.5,max_dist=50,blacklist=False)
from itertools import chain
import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.datasets import fetch_20newsgroups



def sentencewise(all_data):
    all_sentences = []

    for idx, data in enumerate(all_data):
        print(f"Sentence Tokenization: {idx}/{len(all_data)}", end="\r")
        same_data_sentences = []
        sentences = sent_tokenize(data)
        for sentence in sentences:
            words = sentence.split(" ")
            if len(words) <= 2:
                continue
            else:
                sentence = re.sub("^I ", f"Sender{idx} ", sentence)
                sentence = re.sub("^You ", f"Receiver{idx} ", sentence)
                sentence = re.sub("I have |I\'ve ", f"Sender{idx} has", sentence)
                sentence = re.sub("I'm | i'm ", f"Sender{idx} is", sentence)
                sentence = re.sub(" I | i | me ", f" Sender{idx} ", sentence)
                sentence = re.sub(" You | you ", f" Receiver{idx} ", sentence)
                sentence = re.sub("My ", f"Sender{idx}'s ", sentence)
                sentence = re.sub(" my ", f" Sender{idx}'s ", sentence)
                sentence = re.sub("I\'|i'", f" Sender{idx}'", sentence)
                sentence = re.sub(" ?[Yy]ou have", f" Receiver{idx} has ", sentence)
                sentence = re.sub(" ?[Yy]ou've ", f" Receiver{idx} has ", sentence)

                sent = nlp(sentence)
                dependencies = [i.root.dep_ for i in sent.noun_chunks]
                if ("nsubj" in dependencies) and (("dobj" in dependencies) or ("pobj" in dependencies)):
                    if sent._.has_coref:
                        sentence = sent._.coref_resolved
                    same_data_sentences.append(sentence.strip())


                else:
                    continue

        all_sentences.append(same_data_sentences)


    return all_sentences


def get_and_process_data():
    data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
    targets = data['target_names']
    del data['DESCR'], data['target_names']
    df = pd.DataFrame.from_dict(data)
    df['target'] = [targets[i] for i in df.target]

    new_data = []
    filenames = []
    targets = []
    for idx, d in enumerate(data['data']):
        try:
            new_data.append(re.sub('[^A-Za-z0-9\.\!\?\']+', ' ', re.split(r"Lines: \d+(\n*)", d)[-1]).strip())
            filenames.append(data['filenames'][idx])
            targets.append(data['target'][idx])
        except Exception as E:
            print(E, idx)

    df['data'] = new_data
    df['filename'] = filenames
    df['target'] = targets

    del new_data, filenames, targets
    df['processed'] = sentencewise(df['data'])

    return df


def modify_relation(relation, text):
    relation = relation.split(" ")
    loc1 = text.find(relation[0])
    loc2 = text.find(relation[-1]) + len(relation[-1])
    if loc1 > loc2:
        return " ".join(relation)
    return text[loc1:loc2]


def easy_extraction(doc, modify_relations=False):

    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)

    triples = []
    #     relations=[]
    #     all_entities=[]

    for ent in doc.ents:
        preps = [prep for prep in ent.root.head.children if prep.dep_ == "prep"]
        for prep in preps:
            for child in prep.children:
                entities = (ent.text, child.text)
                relation = f"{ent.root.head} {prep}"
                if modify_relations:
                    if relation in doc.text:
                        pass
                    else:
                        relation = modify_relation(relation, doc.text)

                #                         try:
                #                             connector=relations[-1] + " " + all_entities[-1][-1]
                #                             if relations[-1] in connector:
                #                                 relation=relation.replace(" ".join(connector.split(" ")[1:]), "")
                #                         except:
                #                             pass

                #                     all_entities.append(entities)
                #                     relations.append(relation)

                triples.append((entities[0], relation, entities[1]))

    return triples


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, max_len=128, model_name="bert", task="qa"):
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.max_len = max_len
        try:
            processed = df.processed.map(eval)
        except:
            processed = df.processed.map

        self.sentences = list(chain.from_iterable(processed))
        row_number = []
        for idx, item in enumerate(df.num_sentences.values):
            row_number.extend([idx] * item)
        self.row_number = row_number

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        data = self.sentences[idx]
        tokenized = self.tokenizer.encode(data)
        input_ids = tokenized.ids
        offsets = tokenized.offsets

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "offsets": offsets,
            "data": data,
        }