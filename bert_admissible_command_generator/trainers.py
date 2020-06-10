
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from transformers import AdamW, BertConfig,  BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import numpy as np 
import sys
import os
import glob
import json
import copy 
import itertools
import re

import wandb
import logger

from datasets import AdmissibleCommandsClassificationDataset
from models import AdmissibleCommandGenerator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdmissibleCommandTrainer:
    def __init__(self,params):
        
        self.template_coeff = params.template_coeff
        self.object1_coeff = params.object1_coeff
        self.object2_coeff = params.object2_coeff
        self.max_grad_norm = params.max_grad_norm
        self.verbose = params.verbose
        self.print_freq = params.print_freq
        self.checkpoint_dir = params.checkpoint_dir
        self.experiment_name = params.experiment_name
        self.num_train_epochs = params.end_epoch - params.start_epoch
        self.start_epoch = params.start_epoch
        self.end_epoch = params.end_epoch
        self.batch_size = params.batch_size

        self.template2id = eval(open(params.template_file, 'r').read())
        self.id2template = {v: idx for idx, v in self.template2id.items()}
    
        self.object2id = eval(open(params.object_file , 'r').read())
        self.id2object = {v: idx for idx, v in self.object2id.items()}

        wandb.init(project=params.project_name, name = params.experiment_name, config=params, resume= True if params.load_checkpoint else False)


        self.dataset = AdmissibleCommandsClassificationDataset( data_file = params.data_file,
                                                                template2id = self.template2id,
                                                                object2id = self.object2id,
                                                                max_seq_length = params.max_seq_len,
                                                                bert_model_type = params.bert_model_type)


        self.command_generator = AdmissibleCommandGenerator(    num_templates= len(self.template2id),
                                                                num_objects = len(self.object2id),
                                                                embedding_size = params.embedding_size,
                                                                state_hidden_size = params.bert_hidden_size,
                                                                state_transformed_size = params.bert_transformation_size).to(device)

        if params.load_checkpoint:
            checkpoint_path = '{}/{}/Epoch{}/'.format(params.checkpoint_dir, params.experiment_name, params.start_epoch - 1)

            self.bert = BertModel.from_pretrained(checkpoint_path).to(device)
            self.dataset.tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

            with open(checkpoint_path + 'checkpoint.pth', "rb") as f:
                model_dict = torch.load(f, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                self.command_generator.load_state_dict(model_dict['ac_model'])

            train_indices   =  np.loadtxt(params.log_dir + '/' + params.experiment_name + "/train_idx.txt").astype(int)
            val_indices     =  np.loadtxt(params.log_dir + '/' + params.experiment_name + "/val_idx.txt").astype(int)
            test_indices    =  np.loadtxt(params.log_dir + '/' + params.experiment_name + "/test_idx.txt").astype(int)

        else:
            self.bert = BertModel.from_pretrained(params.bert_model_type).to(device)    

            train_indices, val_indices = self.get_train_valid_test_split(len(self.dataset), 0.2)
            test_indices = np.random.choice(train_indices, int(len(train_indices)*0.2) , replace=False)

            save_dir = params.log_dir + '/' + params.experiment_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(save_dir + "/train_idx.txt" ,np.array(train_indices))
            np.savetxt(save_dir + "/val_idx.txt"  ,np.array(val_indices))
            np.savetxt(save_dir + "/test_idx.txt"  ,np.array(test_indices))

        #for debugging
        # train_indices = train_indices[:1*self.batch_size]
        # val_indices = val_indices[:1*self.batch_size]
        # test_indices = test_indices[:1]

        print("Number of Datapoints Train: {}, Val: {}, Test: {}".format(len(train_indices),len(val_indices),len(test_indices)))

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        self.train_dataloader = DataLoader(self.dataset, batch_size=params.batch_size, sampler=train_sampler, drop_last =True)
        self.valid_dataloader = DataLoader(self.dataset, batch_size=params.batch_size, sampler=valid_sampler, drop_last =True)
        self.test_dataloader  = DataLoader(self.dataset, batch_size=params.batch_size, sampler=test_sampler, drop_last =True)

        t_total = len(self.train_dataloader) // params.gradient_accumulation_steps * self.num_train_epochs
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters =  [
            {"params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": params.weight_decay,},
            {"params": [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": self.command_generator.parameters()}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=params.lr, eps=params.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=params.warmup_steps, num_training_steps=t_total)

        if params.load_checkpoint:
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.scheduler.load_state_dict(model_dict['scheduler'])
            
        self.BCE = nn.BCEWithLogitsLoss()
        self.o1_WeightedBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.))
        self.o2_WeightedBCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))
        
        self.configure_logger(log_dir = params.log_dir,experiment_name = params.experiment_name)

    def calc_f_score(self,trans):
        
        logits_t, logits_o1, logits_o2, template_targets,o1_targets, o2_targets = trans

        t_probs = torch.ge(torch.sigmoid(logits_t), 0.5).float().clone().detach().cpu().numpy()
        o1_probs =  torch.ge(torch.sigmoid(logits_o1), 0.5).float().clone().detach().cpu().numpy()
        o2_probs =  torch.ge(torch.sigmoid(logits_o2), 0.5).float().clone().detach().cpu().numpy()
        template_targets = template_targets.cpu().numpy()
        o1_targets = o1_targets.cpu().numpy()
        o2_targets = o2_targets.cpu().numpy()
      
        t_fscore, o1_fscore, o2_fscore = 0,0,0
        t_precision, o1_precision, o2_precision = 0,0,0
        t_recall, o1_recall, o2_recall = 0,0,0

        t_precision, t_recall, t_fscore,_ = precision_recall_fscore_support(template_targets,t_probs,average = 'weighted')

        N = self.batch_size
        for b in range(N):
            o1_p, o1_r, o1_f,_  = precision_recall_fscore_support(o1_targets[b],o1_probs[b],average = 'weighted',  zero_division=1)
            o1_fscore += o1_f
            o1_precision += o1_p
            o1_recall += o1_r

            #average o2 f score over templates and batches
            o2_p, o2_r, o2_f = 0,0,0
            for t in range(len(self.template2id)):
                temp_p, temp_r, temp_f,_  = precision_recall_fscore_support(o2_targets[b][t],o2_probs[b][t],average = 'weighted', zero_division=1)
                o2_p += temp_p
                o2_r += temp_r
                o2_f += temp_f

            #divide by number of templates
            o2_fscore += (o2_f / len(self.template2id))
            o2_precision += (o2_p/ len(self.template2id))
            o2_recall += (o2_r / len(self.template2id))

            
        #divide by batch size
        o1_fscore, o1_precision, o1_recall = o1_fscore/N, o1_precision/N, o1_recall /N
        o2_fscore, o2_precision, o2_recall = o2_fscore/N, o2_precision/N, o2_recall/N
        return  { 
                    'template' : [t_fscore, t_precision, t_recall],
                    'o1'       : [o1_fscore, o1_precision, o1_recall],
                    'o2'       : [o2_fscore, o2_precision, o2_recall]
                }

    def train(self):
        
        self.optimizer.zero_grad()
        for epoch in range(self.start_epoch,self.end_epoch):
            self.bert.train()
            t_fscore, o1_fscore, o2_fscore = 0,0,0
            t_precision, o1_precision, o2_precision = 0,0,0
            t_recall, o1_recall, o2_recall = 0,0,0
            t_loss,obj1_loss,obj2_loss = 0,0,0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc = "Training Epoch {}".format(epoch))):
                batch = tuple(t.to(device) for t in batch)
                state, template_targets, o1_template_targets, o2_o1_template_targets  = batch

                input_ids = state[:,0,:]
                input_mask = state[:,1,:]
                bert_outputs = self.bert(   input_ids,
                                        token_type_ids=None,
                                        attention_mask=input_mask
                                    )
                #pooled output of bert
                encoded_state = bert_outputs[1]  #(batch, 768)
                template_logits, o1_template_logits, o2_o1_logits = self.command_generator(encoded_state)
                
                template_loss   = self.template_coeff   * self.BCE(template_logits, template_targets.to(device))
                o1_loss         = self.object1_coeff    * self.o1_WeightedBCE(o1_template_logits, o1_template_targets.to(device))
                o2_loss         = self.object2_coeff    * self.o2_WeightedBCE(o2_o1_logits, o2_o1_template_targets.to(device))
                loss = (template_loss + o1_loss + o2_loss)
              
                trans = template_logits, o1_template_logits, o2_o1_logits, template_targets, o1_template_targets, o2_o1_template_targets
                metrics = self.calc_f_score(trans)

                t_loss += template_loss.item()          
                obj1_loss += o1_loss.item()             
                obj2_loss += o2_loss.item()           

                t_fscore += metrics['template'][0]     
                t_precision += metrics['template'][1]   
                t_recall += metrics['template'][2]      

                o1_fscore += metrics['o1'][0]          
                o1_precision += metrics['o1'][1]       
                o1_recall += metrics['o1'][2]           

                o2_fscore += metrics['o2'] [0]          
                o2_precision += metrics['o2'][1]        
                o2_recall += metrics['o2'][2] 

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.command_generator.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.bert.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if self.verbose and step % self.print_freq == 0:
                    sys.stdout.flush()
                    train_stats = "\n\tTrain Stats:\n\tStep: {} Template Loss: {} O1 Loss: {}, O2 Loss: {}\n\t\tTEMPLATE FPR: {}\n\t\tO1 FPR: {}\n\t\tO2 FPR: {}\n".format(\
                        step, template_loss/(step + 1),obj1_loss/(step + 1),obj2_loss/(step + 1),\
                        [t_fscore/(step + 1),t_precision/(step + 1), t_recall/(step + 1) ],\
                        [o1_fscore/(step + 1),o1_precision/(step + 1),o1_recall/(step + 1)],\
                        [o2_fscore/(step + 1), o2_precision/(step + 1), o2_recall/(step + 1)])
                    print(train_stats)
                    self.log(train_stats)
                   
                
            eval_metrics = self.eval()
           
            sys.stdout.flush()
            print("EPOCH: ", epoch)
            train_stats = "\n\tTrain Stats:\n\tStep: {} Train Loss: {} O1 Loss: {}, O2 Loss: {}\n\t\tTEMPLATE FPR: {}\n\t\tO1 FPR: {}\n\t\tO2 FPR: {}\n".format(\
                            step, template_loss/(step + 1),obj1_loss/(step + 1),obj2_loss/(step + 1),\
                            [t_fscore/(step + 1),t_precision/(step + 1), t_recall/(step + 1) ],\
                            [o1_fscore/(step + 1),o1_precision/(step + 1),o1_recall/(step + 1)],\
                            [o2_fscore/(step + 1), o2_precision/(step + 1), o2_recall/(step + 1)])
            eval_stats = "\n\tVal Stats:\n\t\tTEMPLATE FPR: {}\n\t\tO1 FPR: {}\n\t\tO2 FPR: {}\n".format(eval_metrics['template'],eval_metrics['o1'],eval_metrics['o2'])
            print(train_stats)
            print(eval_stats)

            self.log(train_stats)
            self.log(eval_stats)

            wandb.log({
                'Template Loss': template_loss/(step + 1),
                 'O1 Loss' : obj1_loss/(step + 1), 
                 'O2 Loss':obj2_loss/(step + 1),
                 'Train Template F Score': t_fscore/(step + 1),
                 'Train O1 F Score' : o1_fscore/(step + 1), 
                 'Train O2 F Score': o2_fscore/(step + 1),
                 'Val Template F Score': eval_metrics['template'][0],
                 'Val O1 F Score' : eval_metrics['o1'][0], 
                 'Val O2 F Score': eval_metrics['o2'][0],
                
            })

            checkpoint = { 
                'epoch': epoch,
                'ac_model': self.command_generator.state_dict(),
                'state_encoder': self.bert.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                }

            save_dir = '{}/{}/Epoch{}/'.format(self.checkpoint_dir, self.experiment_name, epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(checkpoint, save_dir + "checkpoint.pth")
            self.bert.save_pretrained(save_dir)
            self.dataset.tokenizer.save_pretrained(save_dir)
                   
    def eval(self):
        self.bert.eval()
        t_fscore, o1_fscore, o2_fscore = 0,0,0
        t_precision, o1_precision, o2_precision = 0,0,0
        t_recall, o1_recall, o2_recall = 0,0,0
        for step, batch in enumerate(tqdm(self.valid_dataloader,desc = "Evaluation")):  
            batch = tuple(t.to(device) for t in batch)
            state, template_targets, o1_template_targets, o2_o1_template_targets = batch  
            with torch.no_grad():
                input_ids = state[:,0,:]
                input_mask = state[:,1,:]
                bert_outputs = self.bert(   input_ids,
                                        token_type_ids=None,
                                        attention_mask=input_mask
                                    )
                #pooled output of bert
                encoded_state = bert_outputs[1]  #(batch, 768)
                template_logits, o1_template_logits, o2_o1_logits = self.command_generator(encoded_state)

            trans = template_logits, o1_template_logits, o2_o1_logits, template_targets, o1_template_targets, o2_o1_template_targets
           
            metrics = self.calc_f_score(trans)

            t_fscore += metrics['template'][0]     
            t_precision += metrics['template'][1]   
            t_recall += metrics['template'][2]      

            o1_fscore += metrics['o1'][0]          
            o1_precision += metrics['o1'][1]       
            o1_recall += metrics['o1'][2]           

            o2_fscore += metrics['o2'] [0]          
            o2_precision += metrics['o2'][1]        
            o2_recall += metrics['o2'][2]

        eval_metrics = {    
                        'template':  [t_fscore/(step + 1),t_precision/(step + 1), t_recall/(step + 1) ],
                        'o1':  [o1_fscore/(step + 1),o1_precision/(step + 1),o1_recall/(step + 1)],
                        'o2' : [o2_fscore/(step + 1), o2_precision/(step + 1), o2_recall/(step + 1)]
                         } 
        return eval_metrics
    
    def test(self):
        pass
    
    def configure_logger(self,log_dir,experiment_name):
        logger.configure(log_dir + "/"+experiment_name, format_strs=['log'])
        self.log = logger.log

    def get_train_valid_test_split(self,dataset_size, split):
        indices = list(range(dataset_size))
        split = int(np.floor(split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices
