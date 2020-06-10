from torch.utils.data import Dataset
import json
import numpy as np
from transformers import BertTokenizer
import torch
import itertools

class EntityRelationBERTClassificationDataset(Dataset):
    """
    returns batch of state and list of ents/relations present in
    """
    def __init__(self, file,ent2id,rel2id, max_seq_length, model_type):
        with open(file) as json_file:
            self.data = json.load(json_file)
        
        self.ent2id = eval(open(ent2id, 'r').read())
        self.rel2id = eval(open(rel2id, 'r').read())

        self.entity_pairs_ids = list(itertools.product(np.arange(len(self.ent2id)), np.arange(len(self.ent2id))))

        self.tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=False)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]

        x = '[CLS] ' + datapoint['state'] + ' [SEP]'
        x = self.tokenizer.tokenize(x)
        input_ids = self.tokenizer.convert_tokens_to_ids(x)
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_mask = [1] * len(input_ids)
        input_mask += padding
        input_ids += padding
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)

        # segment_mask = [0] * len(input_ids)
        # segment_mask += padding

        ent1_y = torch.tensor(np.zeros((len(self.ent2id),1)))
        ent2_y = torch.tensor(np.zeros((len(self.ent2id),1)))
        rel_y = torch.tensor(np.zeros(len(self.entity_pairs_ids)))
        entities1 = set()
        entities2 = set()

        for fact in datapoint['facts']:
           
            idx = self.entity_pairs_ids.index( (self.ent2id[fact[0]],self.ent2id[fact[1]]))
            #y = torch.tensor(np.zeros(len(self.rel2id)))
            #y[self.rel2id[fact[2]]] = 1
            rel_y[idx] = self.rel2id[fact[2]]

            ent1_y[self.ent2id[fact[0]]] = 1
            ent2_y[self.ent2id[fact[1]]] = 1
   

        return input_ids,input_mask,ent1_y,ent2_y,rel_y