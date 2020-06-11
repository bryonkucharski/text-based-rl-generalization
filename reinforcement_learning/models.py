import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import random
import itertools

import time

from graph_models import StackedRelationalGraphConvolution, GAT
from utils import masked_log_softmax, masked_softmax

import networkx as nx

class CommandEncoder(nn.Module):
    def __init__(self,input_size, embedding_size,device):
        super(CommandEncoder, self).__init__()
        self.hidden_size = embedding_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)#, batch_first=True)
        self.embedding  = nn.Embedding(input_size, embedding_size, padding_idx  = 0)
        self.device = device

    def forward(self, command_rep):
        '''
            command_rep:    batch x num_actions x max_action_len
                            id of each word for each admissible action per batch
        '''
        # batch, num_actions, max_action_len = command_rep.size()
        # command_rep = command_rep.view(batch*num_actions,max_action_len).to(device) #batch*num_action x max_action_len
        # embeds = self.embedding(command_rep) #batch*num_action x max_action_len x embedding_dim
        # encoded_actions, hidden = self.gru(embeds, hidden) #encoded_actions: #batch*num_actions x max_action_len x hidden

        # #take last timestep for each action
        # output = encoded_actions[:,0,:].view(batch,num_actions,-1) #batch x num_actions x hidden

        #return output, hidden
        
        
        batch,num_actions,max_action_len = command_rep.size()
        command_rep = command_rep.view(batch*num_actions,max_action_len).to(self.device )

       
        # Pack the padded batch of sequences
        #this will return the number of words per action. Since the admissible commands are padded with 0, some of the lenghts will need to be 0
        lengths = torch.tensor([torch.nonzero(n)[-1] + 1 if len(np.nonzero(n)) > 0 else torch.Tensor([0]) for n in command_rep], dtype=torch.long).to(self.device )
       

        #PyTorch pack_padded doesnt accept 0 length sequences. This will set the 0 indices equal to 1. the masked_fill call later will mask out these indices.
        #see https://github.com/pytorch/pytorch/issues/4582
        clamped_lengths = lengths.clamp(min=1, max=10)
        
        embedded = self.embedding(command_rep).permute(1,0,2) # T x Batch x EmbDim
        
        hidden = self.initHidden(batch*num_actions)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, clamped_lengths, enforce_sorted=False)

        output, hidden = self.gru(packed, hidden)

        #fill in masked indices with 0 - these are filler admissible commands
        hidden = hidden.masked_fill_((lengths == 0).view(-1, 1), 0).view(batch,num_actions,-1)

        return hidden

    def initHidden(self, size):
        return torch.zeros(1, size, self.hidden_size).to(self.device)

class Actor(nn.Module):
    def __init__(self, tw_vocab_size,embedding_size,state_hidden_size, max_action_len,device,use_gumbel_softmax):
        super(Actor, self).__init__()
        self.max_action_len = max_action_len
        self.use_gumbel_softmax=use_gumbel_softmax
        self.device = device

        self.command_encoder = CommandEncoder(  input_size = tw_vocab_size,
                                                embedding_size = embedding_size,
                                                device = device)

        self.action_scorer = nn.Linear( embedding_size + state_hidden_size,1)
        
        #self.action_scorer = nn.Linear( 371,1) #recipe1
        #self.action_scorer = nn.Linear( 493,1) #recipe3
        #self.action_scorer = nn.Linear( 1178,1)
        #self.action_scorer = nn.Linear( 224,1)

    def forward(self,encoded_state,candidate_action_reps, act_mask, eval_mode=None):
        '''
        
        Score each admissible action and sample one

        encoded_state:              batch x hidden
                                    encoded text description from textworld

        candidate_action_reps:      batch x num_admisisble_actions x max_action_len
                                    list size of batch of numpy arrays. Each numpy array is num_admisisble_actions x max_action_len  

        act_mask:                   Torch Tensor batch x num_actions
                                    binary mask 1 represents an admissible action, 0 represents a padded action      
        '''

        act_rep_tensor = torch.LongTensor(candidate_action_reps).to(self.device)
        batch, num_actions, _ = act_rep_tensor.size()

        #if using BERT
        #batch, num_actions, _ = encoded_act.size()

        encoded_act = self.command_encoder(act_rep_tensor) 
        encoded_act = encoded_act.unsqueeze(1).view(batch,num_actions,-1) # batch x num_admisisble_actions x hidden
 
        #concat admissibble commands with state
        encoded_state = encoded_state.unsqueeze(1).expand(-1,num_actions,-1)    #batch x num_actions x hidden
        #assert encoded_state.sum() == 0

        encoding = torch.cat([encoded_state, encoded_act ],dim=2) #batch x num_actions x 2*hidden
        #encoding = torch.cat([encoded_state, encoded_act ],dim=2)

        #zero out all actions that are fillers
        encoding = (act_mask.unsqueeze(-1).expand(-1,-1,encoding.shape[-1]) * encoding)  #batch x num_actions x 2*hidden

        #score each action and sample
        logits = self.action_scorer(encoding).squeeze(-1)                        #batch x num_actions
        
        #masked softmax
        probs = masked_softmax(logits, act_mask,dim=1)            #batch x num_actions

        if self.use_gumbel_softmax:
            #Gumbel-max for action sampling
            u = torch.rand(probs.size(), dtype=probs.dtype)
            idxs = torch.argmax(probs.cpu() - torch.log(-torch.log(u)), dim=-1)
        else:
            idxs = probs.multinomial(num_samples=1)  #batch x 1

        return idxs, logits

    def gumbel_max_action_sample(self, logits):
        '''
            https://arxiv.org/pdf/1611.01144.pdf
        '''
        probs = torch.softmax(logits,dim=-1)
        u = torch.rand(probs.size(), dtype=logits.dtype)
        idx = torch.argmax(probs - torch.log(-torch.log(u)), dim=-1)
        return idx

class KGEncoder(nn.Module):
    def __init__(self,  batch_size,ent_emb_size,ent_vocab,ent_vocab_rev,rel_emb_size, rel_vocab,rel_vocab_rev,output_dim,dropout_ratio):
        super(KGNetwork, self).__init__()
       
        self.dropout_ratio = dropout_ratio
        self.vocab_ent = ent_vocab
        self.vocab_ent_rev = ent_vocab_rev
        self.vocab_rel = rel_vocab
        self.vocab_rel_rev = rel_vocab_rev
        self.entity_embedding   = nn.Embedding(len(self.vocab_ent), ent_emb_size)
        self.relation_embedding = nn.Embedding(len(self.vocab_rel), rel_emb_size)

        #if GAT
        # self.kg_encoder = GAT(ent_emb_size, 3, dropout_ratio, 0.2, 1)
        # self.state_ent_emb = nn.Embedding(len(self.vocab_ent), ent_emb_size)
        # self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, output_dim)

        #if rgcn
        self.kg_encoder = StackedRelationalGraphConvolution(entity_input_dim = ent_emb_size,
                                                            relation_input_dim = rel_emb_size,
                                                            num_relations = len(self.vocab_rel),
                                                            hidden_dims = [100,100,100,100,100],
                                                            num_bases = 3,
                                                            use_highway_connections=True,
                                                            dropout_rate=0.0)
    
        self.fc = nn.Linear(len(self.vocab_ent) * 64 * 1, output_dim)
        self.batch_size = batch_size

        self.last_facts_seen = [None] * self.batch_size
        

    def extract_GT_facts(self,infos):
        facts = []
        for i in range(len(infos)):
            facts_seen = process_facts(self.last_facts_seen[i], infos[i]["game"], infos[i]["facts"], infos[i]["last_action"], None)
            filtered_facts = serialize_facts(facts_seen)
            self.last_facts_seen[i] = facts_seen
            facts.append(filtered_facts)

        return facts

    def get_adj_with_rel(self,facts):
        adj_matrix = np.zeros((len(facts), len(self.vocab_rel), len(self.vocab_ent), len(self.vocab_ent)), dtype="float32")
        for i, current_facts in enumerate(facts):
            for fact in current_facts:
                u,v,r = fact
                assert u in self.vocab_ent_rev, u + " is not in node vocab"
                assert v in self.vocab_ent_rev, v + " is not in node vocab"
                assert r in self.vocab_rel_rev, r + " is not in relation vocab"
                u_idx = self.vocab_ent_rev[u]
                v_idx = self.vocab_ent_rev[v]
                r_idx = self.vocab_rel_rev[r]
                adj_matrix[i][r_idx][u_idx][v_idx] = 1.0

        return torch.FloatTensor(adj_matrix).to(device)

    def get_adj_no_rel(self):
        ret = []
        adj_matrix = np.zeros((len(self.vocab_ent), len(self.vocab_ent)))

        for u, v in self.graph_state.edges:
            
            if u in self.vocab_ent_rev.keys():
                u_idx = self.vocab_ent_rev[u]
                if v in self.vocab_ent_rev.keys():
                    v_idx = self.vocab_ent_rev[v]
                else:
                    print("Error in get_state_kge: v not in vocab",v)
                    break
            else:
                print("Error in get_state_kge: u not in vocab",u)       
                break

            adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret)), adj_matrix
    
    def extract_mask_ids(self,facts):
        '''
        input:  batch size x len facts
        output: binary list batch size x num_entity
        for each batch, 1 represents entity is present in current graph, 0 if not
        '''
        mask_ids = []
        for i, fact in enumerate(facts):
            edges = [(f[0] ,f[1]) for f in fact]
            ents  = set(list(itertools.chain(*edges)))

            mask_id = torch.LongTensor([0] * len(self.vocab_ent))
            current_ent_ids = torch.LongTensor([self.vocab_ent_rev[ent] for ent in ents])
            mask_id[current_ent_ids] = 1
            mask_ids.append(mask_id)
        
        mask_ids = torch.stack(mask_ids)
        return mask_ids

    def forward(self,infos):
        #return self.forward_GAT(infos)
        return self.forward_RGCN(infos)

    def forward_RGCN(self,infos):
        """
        infos: list batch size x infos for each env
        """

        ###extract graph###
        facts = self.extract_GT_facts(infos) #list of batch triplets
        #future: self.extract_BERT_facts_from_obs(obs) #obs will be batch x words
        
        #used to constrain action space when decoding objects for commands and
        # for only considering node embeddings of current graph
        mask_ids = self.extract_mask_ids(facts)

        ###encode graph###
        entity_features = self.entity_embedding.weight.float().to(device) # 1 x num_entity x emb_size
        entity_features = entity_features.repeat(self.batch_size, 1, 1) # batch x num_entity x emb_size

        relation_features = self.relation_embedding.weight.float().to(device)  # 1 x num_rel x rel_size
        relation_features = relation_features.repeat(self.batch_size, 1, 1) # batch x num_rel x rel_size

        adj = self.get_adj_with_rel(facts) #batch x num_rel x num_ent x num_ent

        kg_encoding = self.kg_encoder(entity_features, relation_features, adj) # batch x num_entity x ent_hidden_dim
        
        #zero out all embeddings that are from nodes not in current graph
        kg_encoding = kg_encoding * mask_ids.unsqueeze(-1).to(device)

        #kg_encoding = kg_encoding.view(self.batch_size,-1) # batch x num_entity * ent_hidden_dim
        #output = self.fc(kg_encoding) #batch x output_dim

        ###encoding aggregation###
        #take mean all nodes of graph. _mask is to not include summation of nodes that are not in graph
        graph_representations = torch.sum(kg_encoding, -2)  # batch x hid
        _mask = torch.sum(mask_ids, -1).to(device)  # batch
        tmp = torch.eq(_mask, 0).float().to(device)
        _mask = _mask + tmp
        output = graph_representations / _mask.unsqueeze(-1)  # batch x hid
        
        return output, mask_ids


    def forward_GAT(self, infos):
        
        out = []
        mask_ids = []
        
        for i in range(self.batch_size):
            self.updateKG(i, infos)
            node_feats, adj =  self.get_adj_no_rel()
            adj = torch.IntTensor(adj).to(device)
            x = self.kg_encoder(self.state_ent_emb.weight, adj)
            import pdb;pdb.set_trace()
            x = x.view(-1)
            out.append(x.unsqueeze_(0))

            edges = list(self.graph_state.edges)
            ents  = set(list(itertools.chain(*edges)))
            ents.remove('player')

            mask_id = torch.LongTensor([0] * len(self.vocab_ent))
            current_ent_ids = torch.LongTensor([self.vocab_ent_rev[ent] for ent in ents])
            mask_id[current_ent_ids] = 1
            #mask_id = [x for x in list(self.vocab_kge.keys()) if x not in current_ent_ids]
            mask_ids.append(mask_id)
       
        out = torch.cat(out)
        ret = self.fc1(out)
        mask_ids = torch.stack(mask_ids)
        return ret, mask_ids

    def updateKG(self, i,infos):
        '''
        Populate the KG
        '''

        facts_seen = process_facts(self.last_facts_seen[i], infos[i]["game"], infos[i]["facts"], infos[i]["last_action"], None)
        filtered_facts = serialize_facts(facts_seen)
        #print(infos['description'],filtered_facts)
        self.last_facts_seen[i] = facts_seen

        self.graph_state = nx.DiGraph()
        for rule in filtered_facts:
            if rule[0] in self.vocab_ent_rev.keys():
                if rule[1] in self.vocab_ent_rev.keys():
                    self.graph_state.add_edge(rule[0],rule[1], rel=rule[2])
                #else:
                
                    #print("v not in vocab",rule, rule[1])
            # else:
            #     if rule[0] is not 'consumed':
            #         print("u not in vocab",rule, rule[0])

class RNNStateEncoder(nn.Module):
    def __init__(self, batch_size,hidden_size, vocab_size,device):
        super(RNNStateEncoder, self).__init__()

        self.batch_size = batch_size

        self.enc_desc = PackedEncoderRNN(vocab_size, hidden_size,device)
        self.enc_inv = PackedEncoderRNN(vocab_size, hidden_size,device)
        self.enc_ob = PackedEncoderRNN(vocab_size, hidden_size,device)
        self.enc_preva = PackedEncoderRNN(vocab_size, hidden_size,device)
        self.enc_recipe = PackedEncoderRNN(vocab_size, hidden_size,device)

        self.init_hidden(self.batch_size)

        self.fcx = nn.Sequential(
                            nn.Linear(hidden_size * 5,int((hidden_size * 3)/2)),
                            nn.ReLU(),
                            nn.Linear( int((hidden_size * 3)/2) ,hidden_size)
        )

        self.fch = nn.Sequential(
                            nn.Linear(hidden_size * 5,int((hidden_size * 3)/2)),
                            nn.ReLU(),
                            nn.Linear( int((hidden_size * 3)/2) ,hidden_size)
        )

        #self.fch = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, obs, save_hidden = False):

        x_d, h_d = self.enc_desc(   obs[:,0,:], self.h_desc)
        x_i, h_i = self.enc_inv(    obs[:,1,:], self.h_inv)
        x_o, h_o = self.enc_ob(     obs[:,2,:], self.h_ob)
        x_p, h_p = self.enc_preva(  obs[:,3,:], self.h_preva)
        x_r, h_r = self.enc_recipe( obs[:,4,:], self.h_r)

        x = self.fcx(torch.cat((x_d,x_i,x_o, x_p,x_r), dim=1))
        h = self.fch(torch.cat((h_d,h_i,h_o, h_p,h_r), dim=2))
            
        if save_hidden:
            self.h_d = h_d
            self.h_i = h_i
            self.h_r = h_r
            self.h_o = h_o
            self.h_p = h_p

        return x, h

    def flatten_parameters(self):
        self.enc_desc.flatten_parameters()
        self.enc_inv.flatten_parameters()
        self.enc_ob.flatten_parameters()
        self.enc_preva.flatten_parameters()
        self.enc_recipe.flatten_parameters()

    def init_hidden(self, batch_size):

        self.h_desc = self.enc_desc.initHidden(batch_size)
        self.h_inv  = self.enc_inv.initHidden(batch_size)
        self.h_ob =     self.enc_ob.initHidden(batch_size)
        self.h_preva =  self.enc_preva.initHidden(batch_size)    
        self.h_r =      self.enc_recipe.initHidden(batch_size)

    def reset_hidden(self, done_mask_tt):
        '''
        Reset the hidden state of episodes that are done.

        :param done_mask_tt: Mask indicating which parts of hidden state should be reset.
        :type done_mask_tt: Tensor of shape [BatchSize x 1]

        '''

        self.h_desc = done_mask_tt.detach() * self.h_desc
        self.h_inv =  done_mask_tt.detach() * self.h_inv
        self.h_ob = done_mask_tt.detach() * self.h_ob
        self.h_preva = done_mask_tt.detach() * self.h_preva
        self.h_r = done_mask_tt.detach() * self.h_r

    def clone_hidden(self):
        ''' Makes a clone of hidden state. '''
        self.tmp_desc = self.h_desc.clone().detach()
        self.tmp_inv = self.h_inv.clone().detach()
        self.tmp_ob = self.h_ob.clone().detach()
        self.tmp_preva = self.h_preva.clone().detach()
        self.tmp_r = self.h_r.clone().detach()
        
    def restore_hidden(self):
        ''' Restores hidden state from clone made by clone_hidden. '''

        self.h_desc = self.tmp_desc
        self.h_inv = self.tmp_inv
        self.h_ob = self.tmp_ob
        self.h_preva = self.tmp_preva
        self.h_r = self.tmp_r

    def get_hidden(self):
        return self.h_desc.clone().detach(), self.h_inv.clone().detach(), self.h_ob.clone().detach(), self.h_preva.clone().detach(), self.h_r.clone().detach()
    
    def set_hidden(self, desc, inv, ob, preva,r):
        self.h_desc = desc
        self.h_inv = inv
        self.h_ob = ob
        self.h_preva = preva
        self.h_r = r
