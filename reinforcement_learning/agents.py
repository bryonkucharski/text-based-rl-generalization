import time
import math, random
import numpy as np
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import random

from sklearn.metrics import precision_recall_fscore_support

from difflib import SequenceMatcher
import re
import itertools

import gym
import textworld
import textworld.gym
from textworld import EnvInfos, text_utils

import logger
import copy


from models import Actor, FeedbackEncoder, RNNStateEncoder
from transformers import BertModel, BertTokenizer

import sentencepiece as spm
import sys

class CGA2CAgent(nn.Module):
    def __init__(self, bert_hidden_size, feedback_output_size, state_output_size,embedding_size,feedback_tokenizer,max_action_len,max_seq_len, tw_vocab_path, batch_size, finetuned_bert_path,device,use_gumbel_softmax=False):
        super(BERTA2CAgent, self).__init__()
        
        self.max_action_len = max_action_len
        self.max_seq_len = max_seq_len
        self.tw_id2word, self.tw_word2id = self.load_tw_vocab(tw_vocab_path)
        self.device= device
        

        self.feedback_tokenizer = feedback_tokenizer
  
        self.actor = Actor(     device = self.device,
                                tw_vocab_size = len(self.tw_word2id),
                                embedding_size = embedding_size,
                                state_hidden_size = state_output_size,
                                max_action_len = self.max_action_len,
                                use_gumbel_softmax = use_gumbel_softmax)

        self.critic = nn.Linear(state_output_size, 1)

        self.state_encoder = RNNStateEncoder(batch_size = batch_size,hidden_size = feedback_output_size, vocab_size = len(self.feedback_tokenizer),device = self.device)

        # self.feedback_encoder = FeedbackEncoder(    hidden_size = feedback_output_size,
        #                                             batch_size = batch_size,
        #                                             vocab_size = len(self.feedback_tokenizer),
        #                                             device = self.device
        #                                     )

        #fc_input_size = bert_hidden_size + feedback_output_size + 10 #10 for score encoding

        # #load pretrained models
        # self.bert = BertModel.from_pretrained(finetuned_bert_path, output_hidden_states=True)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(finetuned_bert_path)

        fc_input_size = 5 * state_output_size
        # self.desc_mlp = nn.Linear(bert_hidden_size,state_output_size)
        # self.inv_mlp = nn.Linear(bert_hidden_size,state_output_size)
        # self.obs_mlp = nn.Linear(bert_hidden_size,state_output_size)
        # self.prev_a_mlp = nn.Linear(bert_hidden_size,state_output_size)
        # self.recipe_mlp = nn.Linear(bert_hidden_size,state_output_size)

        self.state_mlp = nn.Sequential(
                            nn.Linear(fc_input_size,int(fc_input_size/2)),
                            nn.ReLU(),
                            nn.Linear( int(fc_input_size/2) ,state_output_size)
        )
        
    def flatten_parameters(self):
        self.state_encoder.flatten_parameters()
        self.actor.command_encoder.gru.flatten_parameters()

    def feedback_rep_generator(self, feedback_text):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            feedback_text = feedback_text.replace(rm, '')

        feedback_text = feedback_text.split('|')
        ret = [self.feedback_tokenizer.encode_as_ids('<s>' + f + '</s>') for f in feedback_text]

        return self.pad_sequences(ret, maxlen=self.max_seq_len)

    def encode_feedback(self, feedback_text):
        feedback_rep = torch.LongTensor([self.feedback_rep_generator(f) for f in feedback_text]).to(self.device)
        encoded_obs, encoded_prev_a = self.feedback_encoder(feedback_rep,save_hidden = True)
        return encoded_obs, encoded_prev_a

    def bert_rep_generator(self,bert_text, max_seq_len):
        x = '[CLS] ' + bert_text  + ' [SEP]'
        x = self.bert_tokenizer.tokenize(x)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(x)
        
        input_mask = [1] * len(input_ids) 
        diff = (max_seq_len - len(input_ids))
        if diff > 0:
            padding = [0] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding
        else:
            input_ids = input_ids[:max_seq_len]
            input_mask = input_mask[:max_seq_len]
        
        return input_ids, input_mask

    def encode_bert_pooled(self,bert_text,max_seq_len):
        bert_ids, bert_masks = zip(*[self.bert_rep_generator(t, max_seq_len) for t in bert_text])
        bert_ids = torch.LongTensor(list(bert_ids)).to(self.device)
        bert_masks = torch.LongTensor(list(bert_masks)).to(self.device)
        with torch.no_grad():
            bert_outputs = self.bert(bert_ids,token_type_ids=None, attention_mask=bert_masks)
        return bert_outputs[1]  #pooled output of bert

    def encode_bert_penultimate(self,bert_text, max_seq_len):
        bert_ids, bert_masks = zip(*[self.bert_rep_generator(t, max_seq_len) for t in bert_text])
        bert_ids = torch.LongTensor(list(bert_ids)).to(self.device)
        bert_masks = torch.LongTensor(list(bert_masks)).to(self.device)
        ###
        with torch.no_grad():

            bert_outputs = self.bert(bert_ids,token_type_ids=None, attention_mask=bert_masks)

            #tuple len 13. position 0 are embeddings, 1-13 are 12 bert layers
            embed_and_hidden_states = bert_outputs[2] 

            #first token of second to last BERT layer
            #taking second to last layer because BERT was fine tuned on admissible commands, so second to lasr layer may be better for RL
            penultimate_layer_cls = embed_and_hidden_states[-2][:,0,:]

        ###
        return penultimate_layer_cls
        #return bert_outputs[1]  #pooled output of bert

    def encode_score(self, scores):
        src_t = []
        for scr in scores:
            #fist bit encodes +/-
            if scr >= 0:
                cur_st = [0]
            else:
                cur_st = [1]
            cur_st.extend([int(c) for c in '{0:09b}'.format(abs(scr))])
            src_t.append(cur_st)
        return torch.FloatTensor(src_t).to(self.device)
    
    def act(self, desc_text,inventory_text,observation_text,prev_act_text,recipe_text, scores, predicted_valid_acts = None):
        
        batch = len(desc_text)

        ###IF ENCODING STATE WITH BERT
        # # #encodings
        # desc_encoding       =  self.encode_bert_penultimate(desc_text,max_seq_len = 300)
        # inventory_encoding  =  self.encode_bert_penultimate(inventory_text, max_seq_len=50)
        # observation_ecoding =  self.encode_bert_penultimate(observation_text, max_seq_len=50)
        # prev_a_encoding     =  self.encode_bert_penultimate(prev_act_text, max_seq_len=10)
        # # recipe_encoding     =  self.encode_bert_penultimate(recipe_text, max_seq_len=100)

        # # #features
        # desc_features    = self.desc_mlp(desc_encoding )  
        # inventory_features = self.inv_mlp( inventory_encoding )   
        # observation_features = self.obs_mlp( observation_ecoding )    
        # prev_a_features  = self.prev_a_mlp( prev_a_encoding )
        # # recipe_features  = self.recipe_mlp(recipe_encoding )
        # recipe_features = gt_recipe

        # encodings = torch.cat([desc_features, inventory_features, observation_features, prev_a_features], dim=-1)
        # encoded_state = self.state_mlp(encodings)
    
        ###IF ENCODING STATE WITH LSTM
        state_text = [desc_text[i] + "|" + inventory_text[i] + "|" + "OBS:" + observation_text[i] + "|" + "PREVA: " + prev_act_text[i] + "|" + recipe_text[i] for i in range(batch)]
        state_rep =  torch.LongTensor([self.feedback_rep_generator(f) for f in state_text]).to(self.device)
        encoded_state, _ = self.state_encoder(state_rep)
        #encoded_state = encoded_state * 0
    
        value = self.critic(encoded_state)

        if predicted_valid_acts:

            #encode action strings 
            max_num_actions = max([len(acts) for acts in predicted_valid_acts])

             ###F encoding actions with LSTM###
            candidate_action_reps, action_mask = zip(*[self.action_rep_generator(c, max_num_actions) for c in predicted_valid_acts])
            action_mask = torch.Tensor(list(action_mask)).to(self.device)
            #score each command and sample one
            action_idxs, action_logits = self.actor(encoded_state,candidate_action_reps, action_mask)

            ###IF encoding actions with BERT###
            # encoded_actions = []
            # action_mask = []
            # for i,acts in enumerate(predicted_valid_acts):

            #     if len(acts) < max_num_actions:
            #         act_mask = [1] * len(acts) + [0]*(max_num_actions - len(acts))
            #         acts = acts + ['']*(max_num_actions - len(acts)) #pad with empty actions
                    
            #     else:
            #         act_mask = [1]* len(acts)

            #     bert_action = self.encode_bert(acts,max_seq_len=self.max_action_len)
            #     encoded_actions.append(bert_action)
            #     action_mask.append(act_mask)

            # encoded_actions = torch.stack(encoded_actions)
            # action_mask = torch.Tensor(action_mask).to(self.device)

            # action_idxs, action_logits = self.actor(encoded_state,encoded_actions,score_encoding,recipe_features, action_mask, eval_mode = eval_mode)

            return action_idxs, action_logits, action_mask ,value
        else:
            return value

    def action_rep_generator(self,admissible_actions,max_len_admissible_commands): 
        
        max_len=self.max_action_len
        action_rep = []
        for i,act in enumerate(admissible_actions):
            rep = [self.tw_word2id[w] for w in act.split()] + [0]*(max_len - len(act.split()))
            action_rep.append(rep)
        action_rep = action_rep + [[0]*max_len]*(max_len_admissible_commands - len(action_rep)) #pad with empty actions 
        mask = [1]*len(admissible_actions) + [0]* (max_len_admissible_commands - len(admissible_actions))

        return action_rep, mask

    def pad_sequences(self,sequences, maxlen=None, dtype='int32', value=0.):
        '''
        Partially borrowed from Keras
        # Arguments
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            value: float, value to pad the sequences to the desired value.
        # Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
        '''
        lengths = [len(s) for s in sequences]
        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break
        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            # pre truncating
            trunc = s[-maxlen:]
            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                (trunc.shape[1:], idx, sample_shape))
            # post padding
            x[idx, :len(trunc)] = trunc
        return x
    
    def load_tw_vocab(self,tw_vocab_path):
        vocab = {}
        vocab_rev = {} 
        i = 0
        with open(tw_vocab_path) as f:
            for line in f:
                w = line.strip()
                vocab[i] = w
                vocab_rev[w] = i
                i += 1
        return vocab, vocab_rev
    
