import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertConfig, AdamW#, WarmupLinearSchedule
import itertools
import numpy as np


class BertKGTupleClassiciation(BertPreTrainedModel):
    def __init__(self, config,params): 
        super(BertKGTupleClassiciation, self).__init__(config)

        self.params = params
        self.ent2id = eval(open(params.entity_data_file, 'r').read())
        self.rel2id = eval(open(params.relation_data_file, 'r').read())

        self.entity_pairs = list(itertools.product(self.ent2id.keys(), self.ent2id.keys()))
        self.entity_pairs_ids = list(itertools.product(np.arange(len(self.ent2id)), np.arange(len(self.ent2id))))
        self.e1_ids = torch.from_numpy(np.array(self.entity_pairs_ids)[:,0]).to(params.device)
        self.e2_ids = torch.from_numpy(np.array(self.entity_pairs_ids)[:,1]).to(params.device)
        self.entity_embedding = torch.nn.Embedding(len(self.ent2id), self.params.entity_embedding_dim)
    
        self.bert = BertModel(config)
               
        self.entity_1_classifier = nn.Sequential(
                          
                            nn.Linear(self.params.entity_embedding_dim + self.params.bert_hidden_dim, self.params.entity_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.params.entity_hidden_dim, 1)
                            )
        
        self.entity_2_classifier = nn.Sequential(
                            nn.Linear(self.params.entity_embedding_dim + self.params.bert_hidden_dim, self.params.entity_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.params.entity_hidden_dim, 1)
                            )
            
        if self.params.use_rel_attention:
            self.rel_attention = AttentionLayer(self.params.bert_hidden_dim)
            rel_output = 1
        else:
            rel_output = len(self.rel2id)

        self.relation_classifier = nn.Sequential(
                            nn.Linear(2*self.params.entity_embedding_dim + self.params.bert_hidden_dim, self.params.relation_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.params.relation_hidden_dim,rel_output),
                            )

        self.dropout = nn.Dropout(self.params.dropout_prob)
        self.init_weights()
    
    def forward(self, input_ids, input_mask):

        bert_outputs = self.bert(   input_ids,
                                    token_type_ids=None,
                                    attention_mask=input_mask
                                    )
        
        #(batch, 768)
        state_embed = bert_outputs[1] #pooled output of bert

        ###Entity Classifier###
        #(batch, 76, 768) 
        ent_state_embeds = state_embed.unsqueeze(1).repeat(1, len(self.ent2id), 1)

        #76, 128
        entity_embed = self.entity_embedding.weight
        #import pdb;pdb.set_trace()
        #batch, 76, 128
        entity_embeds = entity_embed.repeat(input_ids.shape[0],1,1)

        #batch,76,(128+768)
        entity_classifier_input =  torch.cat((entity_embeds, ent_state_embeds), 2)
        if self.params.use_dropout:
            entity_classifier_input = self.dropout(entity_classifier_input) 

        #batch, 76, 1
        entity_1_logits = self.entity_1_classifier(entity_classifier_input)
        entity_2_logits = self.entity_2_classifier(entity_classifier_input)
        

        ###Relation Classifier###

        #5776, 128
        e1_embeds = self.entity_embedding(self.e1_ids)
        #5776, 128
        e2_embeds = self.entity_embedding(self.e2_ids)
        #5776, 256
        e1e2_embeds = torch.cat((e1_embeds,e2_embeds),1)

        #batch, 5776, 256
        e1e2_embeds = e1e2_embeds.repeat(input_ids.shape[0],1,1)
       
        if self.params.use_rel_attention:
            #batch, max_seq, 768
            last_bert_layer = bert_outputs[0]

            #batch,len(rel2id),768
            state_embed = self.rel_attention(last_bert_layer)
            
            #batch,5776,len(rel2id),768
            rel_state_embeds = state_embed.unsqueeze(1).repeat(1,len(self.entity_pairs_ids),1,1)
            
            #batch, 5776, len(rel2id),256)
            e1e2_embeds = e1e2_embeds.unsqueeze(2).repeat(1,1,len(self.rel2id),1)
              
            #batch,5776,len(rel2id),(256+768)
            relation_classifier_input = torch.cat((e1e2_embeds,rel_state_embeds),3)

        else:
            #batch, 5776, 768
            rel_state_embeds = state_embed.unsqueeze(1).repeat(1,len(self.entity_pairs_ids),1)
            
            #batch, 5776, (768+ 256)
            relation_classifier_input = torch.cat((e1e2_embeds,rel_state_embeds),2)
        
        if self.params.use_dropout:
            relation_classifier_input = self.dropout(relation_classifier_input)
            
        #batch,5776,10
        relation_logits = self.relation_classifier(relation_classifier_input)
        if self.params.use_rel_attention:
            relation_logits = relation_logits.squeeze(3)


        return entity_1_logits, entity_2_logits, relation_logits

class AttentionLayer(nn.Module):
	
    def __init__(self, attention_size):
        super(AttentionLayer, self).__init__()
        self.attention = self.new_parameter(attention_size, 10)
	
    def new_parameter(self,*size):
        out = nn.Parameter(torch.FloatTensor(*size))	
        torch.nn.init.xavier_normal(out)	
        return out
    
    def forward(self, x_in):
        #import ipdb;ipdb.set_trace()
        attention_score = torch.matmul(x_in, self.attention)#.squeeze()

        attention_sm = torch.softmax(attention_score,dim=1)
        attention_sm = attention_sm.unsqueeze(3)

        #batch, seq_len, dims, 1
        x = x_in.unsqueeze(2)
 
        scored_x = x * attention_sm

        #batch, dim
        condensed_x = torch.sum(scored_x, dim=(1))
        
        return condensed_x
