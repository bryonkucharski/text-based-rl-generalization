import os
import argparse

from trainers import A2CTrainer

import torch.multiprocessing as mp

from joblib import Parallel, delayed
import itertools

import sys
import torch
import pickle       
import random
import traceback
import jericho

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def parse_args():
    parser = argparse.ArgumentParser()
            
    parser.add_argument('--policy_coeff', default = 1, type=float)
    parser.add_argument('--value_coeff', default = 0.5, type = float)
    parser.add_argument('--entropy_coeff', default = 0.01, type = float)
    parser.add_argument('--lr', default = 0.001,type = float)
    parser.add_argument('--gamma',  default = 0.95,    type = float)

    parser.add_argument('--training_steps', default = 50000,type = int)
    parser.add_argument('--training_size', default = 100,type = int)
    parser.add_argument('--training_difficulty', default = 7,type = int)
    parser.add_argument('--experiment_name', default = 'name of current run')
    parser.add_argument('--project_name', default = 'my_project',help ='name of project for wandb logging')

    parser.add_argument('--use_gumbel_softmax', default = False, action='store_true')
    parser.add_argument('--use_gt_commands', default = False, action='store_true',help = 'use ground truth admissible commands from textworld instead of generated commands')
    parser.add_argument('--debug', default = False, action='store_true',help ='dont write to log files if debugging')
    
    parser.add_argument('--batch_size',  default = 8,    type = int)
    parser.add_argument('--embedding_size', default = 100,  type = int)
    parser.add_argument('--max_seq_len',  default = 350,  type = int)
    parser.add_argument('--max_action_len',  default = 10,  type = int)
    parser.add_argument('--log_freq', default = 100, type = int)
    parser.add_argument('--bert_hidden_size', default = 768,  type = int)
    parser.add_argument('--feedback_output_size', default = 200,  type = int)
    parser.add_argument('--state_output_size', default = 200,  type = int)
    parser.add_argument('--eval_freq', default = 2000,  type = int)
    parser.add_argument('--game_sample_freq', default = 2000,  type = int)
    parser.add_argument('--max_walkthrough_steps', default = 200,  type = int)
    parser.add_argument('--train_seed', default = 123,  type = int)


    parser.add_argument('--training_type', default = '')
    parser.add_argument('--bert_model_type', default = 'bert-base-cased')
    parser.add_argument('--tw_vocab_path', default = '../data/vocab.txt')
    parser.add_argument('--games_path', default = '../textworld_games/')
    parser.add_argument('--object_file', default = '../data/cooking_games_entities.txt')
    parser.add_argument('--template_file', default = '../data/template2id.txt')
    parser.add_argument('--log_dir', default = 'logs/')
    parser.add_argument('--sp_path', default = '../spm_models/unigram_8k.model')

    #ac model params
    parser.add_argument('--checkpoint_dir', default = '/mnt/nfs/work1/mfiterau/bkucharski/ac_classifier_checkpoints')
    parser.add_argument('--bert_experiment_name', default = 'v1_full_bert_100embed')
    parser.add_argument('--checkpoint_epoch', default = 19, type=int)
       
    parser.add_argument('--env_step_limit',  default = 200,    type = int)
    parser.add_argument('--max_eval_steps', default = 100,type = int)
    parser.add_argument('--eval_mode', default = 'sampled')
    parser.add_argument('--window_average',  default = 5,    type = int)
    parser.add_argument('--bptt',  default = 8,    type = int)
    parser.add_argument('--grad_clip',  default = 40,    type = float)
    parser.add_argument('--weight_decay',  default =  1e-5,    type = float)
    parser.add_argument('--gpu',  default = 0, type = int,help="define which GPU in CUDA_VISIBLE_DEVICES to run on")

    parser.add_argument('--use_extra_reward', default = False, action='store_true')
    parser.add_argument('--gridsearch', default = False, action='store_true')
    parser.add_argument('--train', default = False, action='store_true')
    parser.add_argument('--no_eval', default = False, action='store_true')
    parser.add_argument('--train_v3', default = False, action='store_true')
    parser.add_argument('--backplay', default = False, action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    #to prevent memory error in 2.2.0
    assert jericho.__version__ == '2.4.0', "This code is designed to be run with Jericho version 2.4.0." 
    
    params = parse_args()
 
    if params.train:
        params.experiment_name = 'multigame_samplefreq:{}_diff:{}_type:{}'.format(params.game_sample_freq, params.training_difficulty,params.training_type)
        trainer = A2CTrainer(params)
        trainer.train()

    elif params.backplay:
        params.experiment_name = 'multigame_samplefreq:{}_diff:{}_type:{}'.format(params.game_sample_freq, params.training_difficulty,params.training_type)
        trainer = BERTA2CTrainer(params)
        trainer.train_backplay()
        
   