import sys
import argparse
from tqdm import tqdm
import numpy as np
import logging
import json 
import copy
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from transformers import AdamW, WarmupLinearSchedule, BertConfig
from torch.optim import Adam, SGD
from models import BertKGTupleClassiciation
from datasets import EntityRelationBERTClassificationDataset


from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

from utils import load_pickle, save_pickle, get_train_valid_test_split, get_eval_metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate_model(valid_loader,model,entity_criterion,relation_criterion, entity_thres = 0.5,relation_thres = 0.5):
    model.eval()

    eval_loss,eval_loss1,eval_loss2,eval_loss3 = 0,0,0,0
    eval_e1_precision, eval_e1_recall, eval_e1_fscore = 0,0,0
    eval_e2_precision, eval_e2_recall, eval_e2_fscore = 0,0,0
    eval_rel_precision,eval_rel_recall, eval_rel_fscore = 0,0,0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    batch_num = 0
    for i, batch in enumerate(tqdm(valid_loader,desc = "Evaluation")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, ent1_y,ent2_y,rel_y  = batch
      
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            entity_1_logits,entity_2_logits, relation_logits = model(input_ids, input_mask)
            loss1 = entity_criterion(entity_1_logits.double(),ent1_y)
            loss2 = entity_criterion(entity_2_logits.double(),ent2_y)
            loss3 = relation_criterion(relation_logits.double(),rel_y)
            eval_loss += (loss1.double() + loss2.double() + loss3.double())/3
            eval_loss1 += loss1
            eval_loss2 += loss2
            eval_loss3 += loss3

        eval_metrics = get_eval_metrics(entity_1_logits,entity_2_logits,relation_logits,ent1_y,ent2_y,rel_y,entity_thres, relation_thres)
        eval_e1_precision += eval_metrics['e1'][0]
        eval_e1_recall += eval_metrics['e1'][1]
        eval_e1_fscore += eval_metrics['e1'][2]
        
        eval_e2_precision += eval_metrics['e2'][0]
        eval_e2_recall += eval_metrics['e2'][1]
        eval_e2_fscore += eval_metrics['e2'][2]

        eval_rel_precision += eval_metrics['rel'][0]
        eval_rel_recall += eval_metrics['rel'][1]
        eval_rel_fscore += eval_metrics['rel'][2]
        nb_eval_steps += 1

    metrics = {
        'e1':  (eval_loss1.item()/nb_eval_steps, eval_e1_precision/nb_eval_steps ,  eval_e1_recall/nb_eval_steps,  eval_e1_fscore/nb_eval_steps),
        'e2':  (eval_loss2.item()/nb_eval_steps, eval_e2_precision/nb_eval_steps ,  eval_e2_recall/nb_eval_steps,  eval_e2_fscore/nb_eval_steps),
        'rel':  (eval_loss3.item()/nb_eval_steps, eval_rel_precision/nb_eval_steps ,  eval_rel_recall/nb_eval_steps,  eval_rel_fscore/nb_eval_steps),    
        'eval_loss':  (eval_loss/nb_eval_steps).item()
    }
    
    return metrics

def train_model(train_dataloader, valid_dataloader,model,optimizer,scheduler, args,class_weight,pos_weight, verbose=True):
    model.train()
    if not args.load_checkpoint: 
        output_file = open(args.output_path+ "/logs/" + args.model_name + ".txt", "w").close() #reset this file
    model.zero_grad()
    metrics = {}
    ent_loss_func = torch.nn.BCEWithLogitsLoss() #Sigmoid layer and BCELoss
    rel_loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   
    for epoch in range(args.start_epoch,args.end_epoch):
        tr_loss, tr_loss1, tr_loss2, tr_loss3 = 0.0, 0.0, 0.0, 0.0
        nb_tr_steps = 0
        train_e1_precision, train_e2_precision, train_rel_precision = 0,0,0
        train_e1_recall, train_e2_recall,train_rel_recall = 0,0,0
        train_e1_fscore, train_e2_fscore,train_rel_fscore = 0,0,0 
        for step, batch in enumerate(tqdm(train_dataloader, desc = "Training Epoch {}".format(epoch))):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, ent1_y,ent2_y,rel_y  = batch
                            
            entity_1_logits,entity_2_logits, relation_logits = model(input_ids, input_mask)
  
            loss1 = ent_loss_func(entity_1_logits.double(),ent1_y)
            loss2 = ent_loss_func(entity_2_logits.double(),ent2_y)
            loss3 = rel_loss_func(relation_logits.double(),rel_y)
               
            loss = loss1.double() + loss2.double() + loss3.double()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps   
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                if scheduler:
                    scheduler.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            tr_loss1 += loss1.item()
            tr_loss2 += loss2.item()
            tr_loss3 += loss3.item()
            nb_tr_steps += 1

            train_metrics = get_eval_metrics(entity_1_logits,entity_2_logits,relation_logits,ent1_y,ent2_y,rel_y,0.5,0.5)
            
            train_e1_precision += train_metrics['e1'][0]
            train_e1_recall += train_metrics['e1'][1]
            train_e1_fscore += train_metrics['e1'][2]
            train_e2_precision += train_metrics['e2'][0]
            train_e2_recall += train_metrics['e2'][1]
            train_e2_fscore += train_metrics['e2'][2]
            train_rel_precision += train_metrics['rel'][0]
            train_rel_recall += train_metrics['rel'][1]
            train_rel_fscore += train_metrics['rel'][2]
            
            if step % args.print_freq == 0:
                sys.stdout.flush()
                output_file = open(args.output_path + "/logs/" + args.model_name + ".txt", "a")
                train_out_string = "\nEpoch: {}, Step: {}\tTrain Loss E1: {}, Train E1 F1: {}, Train E2 Loss: {}, Train E2 F1: {}, Train Rel Loss: {}, Train Rel F1: {}".format(epoch,step,(tr_loss1/nb_tr_steps), (train_e1_fscore/nb_tr_steps), (tr_loss2/nb_tr_steps),(train_e2_fscore/nb_tr_steps), (tr_loss3/nb_tr_steps),(train_rel_fscore/nb_tr_steps) )
                print(train_out_string)
                output_file.write(train_out_string)
                output_file.close()
           
                
        
        eval_metrics = evaluate_model(valid_dataloader,model,ent_loss_func,rel_loss_func)
        
        if verbose: 
            sys.stdout.flush()
            output_file = open(args.output_path + "/logs/" + args.model_name + ".txt", "a")
            train_out_string = "\nEpoch: {}\n\tTrain Loss E1: {}, Train E1 F1: {}, Train E2 Loss: {}, Train E2 F1: {}, Train Rel Loss: {}, Train Rel F1: {}".format(epoch,(tr_loss1/nb_tr_steps), (train_e1_fscore/nb_tr_steps), (tr_loss2/nb_tr_steps),(train_e2_fscore/nb_tr_steps), (tr_loss3/nb_tr_steps),(train_rel_fscore/nb_tr_steps) )
            valid_out_string = "\tValid Loss E1: {}, Valid E1 F1: {}, Valid E2 Loss: {}, Valid E2 F1: {}, Valid Rel Loss: {}, Valid Rel F1: {}".format((eval_metrics['e1'][0]), (eval_metrics['e1'][3]), (eval_metrics['e2'][0]),(eval_metrics['e2'][3]), (eval_metrics['rel'][0]),(eval_metrics['rel'][3]) )
            print(train_out_string)
            print(valid_out_string)
            output_file.write(train_out_string)
            output_file.write("\n"+valid_out_string)
            output_file.close()
            
        
        eval_metrics.update({
            'train_loss_1' : tr_loss1/nb_tr_steps,
            'train_loss_2' : tr_loss2/nb_tr_steps,
            'train_loss_3' : tr_loss3/nb_tr_steps,
            'train_e1_f1'  : train_e1_fscore /nb_tr_steps,
            'train_e2_f1'  : train_e2_fscore /nb_tr_steps,
            'train_rel_f1' : train_rel_fscore/nb_tr_steps
        })
        metrics.update({epoch: eval_metrics})
        path = args.output_path + "models/"+args.model_name + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        save_pickle(path,args.model_name+"_epoch_" + str(epoch), model)   
          
    return model, metrics

def get_args():
    parser = argparse.ArgumentParser()

    #File Paths
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--entity_data_file", default="data/ent2id.txt", type=str, 
                        help="The input ent2id  file (a text file).")
    parser.add_argument("--relation_data_file", default="data/rel2id.txt", type=str, 
                        help="The input rel2id  file (a text file).")
    parser.add_argument("--output_path", default="outputs/", type=str,
                        help="location to save all outputs")
    parser.add_argument("--model_name", default="tuple_extractor", type=str,
                        help="location to save all outputs")

     

    parser.add_argument("--entity_embedding_dim", default=128, type=int, 
                        help="dimensions of entity embeddings")
    parser.add_argument("--entity_hidden_dim", default=128,type=int,
                         help="dimensions of hidden dimensions of both entity MLPS")
    parser.add_argument("--relation_hidden_dim", default=128,type=int,
                         help="dimensions of hidden dimensions of both entity MLPS")

    #Bert Params
    parser.add_argument("--bert_hidden_dim", default=768, type=int, 
                        help="dimensions of output of bert model")           
    parser.add_argument("--bert_model_type", default="bert-base-cased", type=str, 
                        help="type of pretrained bert model")

    #training params
    parser.add_argument("--train_model",action='store_true',
                        help="Train model")
    parser.add_argument("--eval_model",action='store_true',
                        help="eval model")
    parser.add_argument("--load_checkpoint",action='store_true',
                        help="load previously trained model under model_name")
    parser.add_argument("--start_epoch", default=3, type=int,
                        help="epoch to start training. Typically 0 unless loading checkpoint model")
    parser.add_argument("--end_epoch", default=14, type=int,
                        help="last epoch to run") 
    parser.add_argument("--batch_size", default=2, type=int,
                        help="") 
    parser.add_argument("--max_sequence_length", default=400, type=int,
                        help="Max number of tokens for the input to bert model")
    parser.add_argument("--lr", default =  0.1, type=float,
                        help="learning rate")
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="False to fine tune BERT during training, True to keep frozen")
    parser.add_argument("--seed", default=-1, type=int,
                        help="seed for random generators")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="percentage to dropout")
    parser.add_argument("--print_freq", default=2000, type=int,
                        help="how many training steps in between logging")
    parser.add_argument("--use_rel_attention", default=True, type=bool,
                        help="attention for relation classifier")
    parser.add_argument("--use_dropout", default=False, type=bool,
                        help="dropout before linear layers for classifiers")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="seed for random generators")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")#was0.01
    parser.add_argument("--adam_epsilon", default=1e-4, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    return parser.parse_args()
    

if __name__ == '__main__':

    
    args = self.get_args()

    if args.seed >= 0:
        print("SEEDING WITH ", args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)

    dataset = EntityRelationBERTClassificationDataset(  args.train_data_file,
                                                        args.entity_data_file,
                                                        args.relation_data_file,
                                                        args.max_sequence_length,
                                                        args.bert_model_type)
    print("RUNNING ON DEVICE: ",device)
    
    train_indices, val_indices = get_train_valid_test_split(len(dataset), 0.2)
    test_indices = np.random.choice(train_indices, int(len(train_indices)*0.2) , replace=False)
    print("Number of Datapoints Train: {}, Val: {}, Test: {}".format(len(train_indices),len(val_indices),len(test_indices)))

    #for debugging
    #train_indices = train_indices[:1]
    #val_indices = val_indices[:1]
    #test_indices = test_indices[:1]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)
    test_dataloader  = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

    #np.savetxt("train_idx.txt" ,np.array(train_indices))
    #np.savetxt("val_idx.txt"  ,np.array(val_indices))
    #np.savetxt("test_idx.txt"  ,np.array(test_indices))

    args.device = device
    args.num_epochs = args.end_epoch - args.start_epoch

    model = BertKGTupleClassiciation.from_pretrained(args.bert_model_type,params=args ).to(device)    
    if args.load_checkpoint:
        path = args.output_path + "models/"+args.model_name + "/" + args.model_name+"_epoch_" + str(args.start_epoch-1) + ".pkl"
        print("Loading model from: ", path)
        model = load_pickle(path, model)
           
    param_optimizer = list(model.named_parameters())
    t_total = len(train_indices) // args.gradient_accumulation_steps * args.num_epochs
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    if args.train_model:
        class_weight = torch.Tensor([1, 2.2221e+03, 5.4913e+03, 5.6985e+03, 3.3274e+03, 1.1819e+03, 2.3684e+03, 8.8197e+02, 5.4487e+03, 5.6050e+03, 1.8455e+03]).to(device)       
        pos_weight = torch.Tensor([ 2.22112180e+03, 5.49014410e+03, 5.69754342e+03,3.32644187e+03, 1.18088233e+03, 2.36735861e+03, 8.80970311e+02, 5.44783770e+03, 5.60399247e+03, 1.84452082e+03]).to(device)
        #pos_weight = torch.Tensor([ 10, 10 , 10 ,10 , 10, 10, 10, 10, 10, 10]).to(device)
        
        model, metrics = train_model(train_dataloader = train_dataloader,
                            valid_dataloader = valid_dataloader,
                            model = model,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            args = args,
                            class_weight = class_weight,
                            pos_weight = pos_weight
                            )
    
        with open(args.output_path + "metrics/" + args.model_name + '_metrics.json', 'w') as fp:
            json.dump(metrics, fp)
        save_pickle(args.output_path + "models/",args.model_name, model)


    elif args.eval_model:
        
        model = load_pickle(args.output_path + "models/" + args.model_name + ".pkl", model)
        evaluate_model(train_dataloader,model,nn.BCEWithLogitsLoss(),entity_thres=0.5)
      

    
    
