
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertModel, BertTokenizer

import numpy as np 
import sys
import logger
import os
import glob
import json
import copy 
import itertools
import re
import time

from collections import Counter
import wandb

import logger
import random
from tqdm import tqdm

import gym
import textworld
import textworld.gym
from textworld import EnvInfos, text_utils

from agents import CGA2CAgent
from admissible_command_generator.models import AdmissibleCommandGenerator
from evaluate import evaluate_agent
from utils import *
from utils import _SentencepieceProcessor

class A2CTrainer:
    def __init__(self, params):

        #TODO: fix this 
        self.policy_coeff = params.policy_coeff
        self.value_coeff  = params.value_coeff
        self.entropy_coeff = params.entropy_coeff
        self.debug_mode = params.debug
        self.batch_size = params.batch_size
        self.gamma = params.gamma 
        self.bptt = params.bptt
        self.clip = params.grad_clip
        self.log_freq = params.log_freq
        self.window_average = params.window_average
        self.use_gt_commands = params.use_gt_commands
        self.env_step_limit = params.env_step_limit
        self.eval_freq = params.eval_freq
        self.training_steps = params.training_steps
        self.max_eval_steps = params.max_eval_steps
        self.game_sample_freq = params.game_sample_freq
        self.env_step_limit = params.env_step_limit
        self.max_walkthrough_steps = params.max_walkthrough_steps
        self.use_extra_reward = params.use_extra_reward
        self.device = torch.device('cuda:{}'.format(params.gpu) if torch.cuda.is_available() else 'cpu')
        self.debug_index = 1
        self.eval_mode = params.eval_mode
        self.train_seed = params.train_seed
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.max_seq_len = params.max_seq_len
        self.no_eval = params.no_eval

        self.set_seed(self.train_seed)
        finetuned_bert_path = '{}/{}/Epoch{}/'.format(params.checkpoint_dir, params.bert_experiment_name, params.checkpoint_epoch)
        self.experiment_name = params.experiment_name
        
        self.template2id = eval(open(params.template_file, 'r').read())
        self.id2template = {v: idx for idx, v in self.template2id.items()}
    
        self.object2id = eval(open(params.object_file , 'r').read())
        self.id2object = {v: idx for idx, v in self.object2id.items()}

        self.direction2id = {'roasted':0, 'grilled':1, 'fried':2, 'sliced':3, 'chopped':4, 'diced':5,'uncut':6,'raw':7 }

        self.template_size = len(self.template2id)
        self.object_size = len(self.object2id)

    
        sp = _SentencepieceProcessor(params.sp_path)

        self.agent = CGA2CAgent(      device = self.device,
                                        bert_hidden_size = params.bert_hidden_size,
                                        feedback_output_size = params.feedback_output_size,
                                        state_output_size = params.state_output_size,
                                        embedding_size = params.embedding_size,
                                        feedback_tokenizer = sp,
                                        max_action_len = params.max_action_len,
                                        max_seq_len = params.max_seq_len,
                                        tw_vocab_path = params.tw_vocab_path,
                                        batch_size = self.batch_size,
                                        finetuned_bert_path = finetuned_bert_path,
                                        use_gumbel_softmax=params.use_gumbel_softmax).to(self.device)

        
    
        self.command_generator = AdmissibleCommandGenerator( num_templates = len(self.template2id),
                                                        num_objects = len(self.object2id),
                                                        embedding_size = params.embedding_size,
                                                        state_hidden_size = params.bert_hidden_size).to(self.device)
        
                                            
        with open(finetuned_bert_path + 'checkpoint.pth', "rb") as f:
            model_dict = torch.load(f, map_location=self.device)
            self.command_generator.load_state_dict(model_dict['ac_model'])

        if not self.debug_mode:
            wandb.init(project=params.project_name,name = params.experiment_name ,config = params)
            #wandb.watch(self.command_generator, log="all")
            wandb.watch(self.agent, log="all")

        self.env_options = textworld.EnvInfos(command_templates=True,
                                        facts=True,
                                        intermediate_reward = True,
                                        admissible_commands=True,
                                        last_action = True,
                                        game = True,
                                        description=True,
                                        entities=True,
                                        max_score = True,
                                        inventory = True,
                                        won = True,
                                        lost = True,
                                        extras=["recipe","walkthrough", "goal"])


        self.log = configure_logger(log_dir = params.log_dir,experiment_name=params.experiment_name)
       
        self.all_train_games = get_train_games(params.games_path,params.training_size,params.training_difficulty)
        self.all_eval_games  = get_eval_games(params.games_path,params.training_size, params.training_difficulty)

        self.MSE = nn.MSELoss()

        self.a2c_optimizer = optim.Adam(self.agent.parameters(), lr=params.lr,weight_decay = params.weight_decay)

        if params.training_size > 1:

            '''
            Options:

            8 games at once - high turnover - need to ensure each game can properly learn
            4 games at once - good blend between the two?
            1 game at once - maybe do this if games cant learn in other cases
            '''
            #all possible game ids
            self.train_env_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.env_step_limit) for game in self.all_train_games]
            
            #sample batch size amount of train envs
            sampled_train_ids = np.random.randint(len(self.all_train_games), size=self.batch_size)
            self.train_envs = [gym.make(self.train_env_ids[i]) for i in sampled_train_ids]
            self.current_train_ids = sampled_train_ids

            # #sample batch size of the same env
            # idx = np.random.randint(len(self.all_train_games))
            # game = self.all_train_games[idx]
            # game_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.env_step_limit)  for i in range(self.batch_size)]
            # self.train_envs = [gym.make(i) for i in game_ids]
            # self.current_train_ids = [idx] * self.batch_size 

            eval_env_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.max_eval_steps) for game in self.all_eval_games]
            self.all_eval_envs = [gym.make(i) for i in eval_env_ids]

            # sampled_train_ids = np.random.randint(len(self.all_train_games), size=20)
            # self.subset_train_envs = [gym.make(self.train_env_ids[i]) for i in sampled_train_ids]

            # self.diff2_eval_games = self.get_eval_games(params.games_path,20, 2)
            # diff2_eval_env_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.max_eval_steps) for game in self.diff2_eval_games]
            # self.diff2_eval_envs = [gym.make(i) for i in diff2_eval_env_ids]
            
            # self.diff3_eval_games = self.get_eval_games(params.games_path,20, 3)
            # diff3_eval_env_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.max_eval_steps) for game in self.diff3_eval_games]
            # self.diff3_eval_envs = [gym.make(i) for i in diff3_eval_env_ids]

            # self.diff7_eval_games = self.get_eval_games(params.games_path,20, 7)
            # diff7_eval_env_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.max_eval_steps) for game in self.diff7_eval_games]
            # self.diff7_eval_envs = [gym.make(i) for i in diff7_eval_env_ids]


        elif params.training_size == 1:
            
            #get one game
            #idx = np.random.randint(len(self.all_train_games))
            #idx = self.all_train_games.index('twgames/cooking_theme/additional//train_100/difficulty_level_6/tw-cooking-recipe1+take1+open+go12-mNE9uoVkHoYjUr8eF3Wy.z8')
            # diff 8 - tw-cooking-recipe3+take3+open+go6-gjklUVelc72LT7Jehp2m
            #diff 6 - tw-cooking-recipe1+take1+open+go12-mNE9uoVkHoYjUr8eF3Wy
            #diff 2 - tw-cooking-recipe1+take1+cook+open-rBJBskJPCeOqcVDPcGxK.z8
            #diff 4 - tw-cooking-recipe1+take1+open+go6-vMVMtjdEuqpQfbjRUQe2.z8
            
            idx = 0 #theres only one game
            game = self.all_train_games[idx]
            #print(idx,game)
            self.current_train_ids = [0] * self.batch_size
           
            #multiple batches of same game
            game_ids = [textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.env_step_limit)  for i in range(self.batch_size)]

            self.all_train_envs = [gym.make(i) for i in game_ids]
            self.all_eval_envs = self.all_train_envs
            self.train_envs = self.all_train_envs
            self.subset_train_envs = [gym.make(i) for i in game_ids]

    def set_seed(self,seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
    
        episodes               = [0] * self.batch_size
        steps_per_episode      = [0] * self.batch_size
        scores                 = [0] * self.batch_size
        last_scores            = [0] * self.batch_size
        current_training_steps = [0] * self.batch_size
        steps_per_game         = [0] * self.batch_size
        dones                  = [False] * self.batch_size
        last_x_episode_score =  [0] * self.batch_size
        last_x_episode_steps = [self.env_step_limit] * self.batch_size

        all_scores                  = [[] for i in range(self.batch_size)]
        average_scores              = [[] for i in range(self.batch_size)]
        all_steps_per_episode       = [[] for i in range(self.batch_size)]
        average_steps_per_episode   = [[] for i in range(self.batch_size)]
      
        obs, infos = zip(*[env.reset() for env in self.train_envs])

        desc_text,inventory_text,observation_text,recipe_text = get_game_text(obs,infos,self.batch_size)
        prev_a_text = ["None" for i in range(self.batch_size) ]
                                                      
        a2c_transitions = []
        converged = False
        average_batched_scores = 0
        average_batched_steps  = self.env_step_limit
        total_steps = 0
        t = 0
        num_games_trained = self.batch_size
        seen_games = set()
        num_score_check = 0
        num_step_check = 0
        num_eval = 0
        for idx in self.current_train_ids:
            seen_games.add(idx)

        steps_counter = 0
        score_counter = Counter()
        while total_steps <= self.training_steps:
            
            gt_admissible = get_filtered_admissible_commands(infos)

            #generate actions
            if self.use_gt_commands:
                predicted_valid_acts = gt_admissible
            else:
                with torch.no_grad():
                    #import pdb;pdb.set_trace()
                    bert_text = [clean_game_state("DESCRIPTION: "+ infos[i]['description'] + " INVENTORY: "+ infos[i]['inventory']) for i in range(self.batch_size)]
                    bert_encoding = self.agent.encode_bert_pooled(bert_text,self.max_seq_len)

                    #TODO move this out of agent if agent is not going to use BERT
                    logits_t, logits_o1, logits_o2 = self.command_generator(bert_encoding)

                predicted_valid_acts = get_generated_valid_actions(logits_t, logits_o1, logits_o2,self.id2template,self.id2object)

            action_idxs, action_logits,action_mask, value = self.agent.act(desc_text,inventory_text,observation_text,prev_a_text,recipe_text, scores=scores, predicted_valid_acts = predicted_valid_acts)
            action_strs = [predicted_valid_acts[i][action_idxs[i].item()] for i in range(self.batch_size)]
    
            next_obs, scores, dones , next_infos, reward, last_scores = step_multi_env(self.train_envs,action_strs, last_scores)
            steps_per_episode = [steps_per_episode[i] + 1 for i in range(self.batch_size)]

            ### For debugging ###
            sys.stdout.flush()
            current_game = self.all_train_games[self.current_train_ids[self.debug_index]].split("/")[-1]
            curr_obj  = infos[self.debug_index]['game'].metadata['ingredients'][0][0]
            opt_command = infos[self.debug_index]['game'].metadata['ingredients'][0][-1]
            print("Step: {}, Episode Step: {} Game: {} Ing: {} Optimal Command: {}\nScores: {} Rewards: {} Dones: {} Actions: {}\nCurrent Train Envs: {}\nInventory: {}".format(total_steps,\
                 steps_per_episode[self.debug_index],current_game,curr_obj,opt_command, scores[self.debug_index],reward[self.debug_index], dones[self.debug_index], action_strs[self.debug_index],self.current_train_ids,infos[self.debug_index]['inventory']))
            if not self.debug_mode:
                self.log( "Step: {}, Episode Step: {} Game: {}\nScores: {} Rewards: {} Dones: {} Actions: {}\nInventory: {}".format(total_steps, steps_per_episode[self.debug_index],current_game, scores[self.debug_index],reward[self.debug_index], dones[self.debug_index], action_strs[self.debug_index],infos[self.debug_index]['inventory']) )
            

            named_action_scoring_probs = [(s, p.item()) for s,p in zip(predicted_valid_acts[self.debug_index], masked_softmax(action_logits,action_mask,dim=1)[self.debug_index])]
            named_action_scoring_probs.sort(key = lambda x: x[1],reverse=True)
            
            for i in range(len(named_action_scoring_probs[:10])):
                print(named_action_scoring_probs[i])
                if not self.debug_mode:
                    self.log(named_action_scoring_probs[i])
            self.log(" ")
            print()
            #########

            next_desc_text,next_inventory_text,next_observation_text,next_recipe_text = get_game_text(next_obs,next_infos,self.batch_size)
            next_prev_a_text = action_strs

            done_mask =  (~torch.tensor(dones)).float().to(self.device).unsqueeze(1)
            rw = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            self.agent.state_encoder.reset_hidden(done_mask)

            a2c_transitions.append((
                action_idxs,
                action_logits,
                action_mask,
                value,
                rw,
                done_mask
            ))

            #update agent
            if len(a2c_transitions) >= self.bptt:
                self.agent.state_encoder.clone_hidden()
                next_value = self.agent.act(next_desc_text,next_inventory_text,next_observation_text,prev_a_text,next_recipe_text)
                returns, advantages = discount_reward(a2c_transitions, next_value,self.gamma)

                if not self.debug_mode:
                    wandb.log({
                        'Advantage': advantages[-1].median().item(),
                        'Value': value.mean().item()
                    },commit=False)

                loss, policy_loss, value_loss, entropy_loss = update_a2c(a2c_transitions, returns, advantages,\
                                policy_coeff=self.policy_coeff,value_coeff = self.value_coeff, entropy_coeff = self.entropy_coeff)
                del a2c_transitions[:]


                if not self.debug_mode:
                    wandb.log({
                        'A2CLoss': loss.item(),
                        'PolicyLoss': policy_loss.item(),
                        'ValueLoss': value_loss.item(),
                        'EntropyLoss': entropy_loss.item()
                    },commit=False)

                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip)

                self.a2c_optimizer.step()
                self.a2c_optimizer.zero_grad()

                self.agent.state_encoder.restore_hidden() 

            #check if any env is done and reset
            for i, done in enumerate(dones):
                if done:
                    #keep track of stats
                    score = scores[i]
                    step = steps_per_episode[i]
                    all_scores[i].append(score)
                    all_steps_per_episode[i].append(step)
                    
                    episodes[i] += 1 
                    sys.stdout.flush()
                    if i == self.debug_index:
                        print('-' * 10, 'New Episode', '-'*10)
                        print('Batch ID: ', i, 'Episode Score: ', score, 'Episode Steps: ', step)
                        self.log("\n\nEPISODE SCORE: {} EPISODE STEPS: {}\n\n".format(score,step))

                    #reset this env
                    n_obs, n_infos = self.train_envs[i].reset()

                    next_desc_text[i] = clean_game_state("DESCRIPTION: "+ n_infos['description']) 
                    next_inventory_text[i] = clean_game_state("INVENTORY: "+ n_infos['inventory']) 
                    next_observation_text[i] = clean_game_state(n_obs) 
                    next_recipe_text[i] = clean_game_state((n_infos['extra.recipe'] or n_infos['extra.goal']))
                    next_prev_a_text[i] = 'None'
                    next_infos[i] = n_infos


                    steps_per_episode[i] = 0
                    scores[i] = 0
                    dones[i] = False
                    last_scores[i] = 0

            if self.training_size == 1:
                #logging and averaging
                if total_steps % self.log_freq == 0:

                    for i in range(self.batch_size):
                        if len(all_scores[i]) < self.window_average:
                            last_x_episode_score[i] = 0
                            last_x_episode_steps[i] = self.env_step_limit
                        else:
                            last_x_episode_score[i] = np.average(all_scores[i][-self.window_average:])
                            last_x_episode_steps[i] = np.average(all_steps_per_episode[i][-self.window_average:])

                        if np.isnan(last_x_episode_score[i]):
                            last_x_episode_score[i] = 0
                        if np.isnan(last_x_episode_steps[i]):
                            last_x_episode_steps[i] = self.env_step_limit
                        
                
                    #average across all batches
                    all_last_x_episode_scores = np.average(last_x_episode_score)
                    all_last_x_episode_steps  = np.average(last_x_episode_steps)
                    average_scores.append(all_last_x_episode_scores)
                    average_steps_per_episode.append(all_last_x_episode_steps)

                    if not self.debug_mode:
                        wandb.log({
                                "AverageWindowedEpisodeScore": all_last_x_episode_scores,
                                "AverageWindowedStepsPerEpisode": all_last_x_episode_steps,
                        })

            else:
                #convergence check to see if time to change game
                for i in range(self.batch_size):
                    #if ran at least x episodes and got max score over those x episodes
                    score_check = len(all_scores[i]) >= self.window_average and np.average(all_scores[i][-self.window_average:]) == infos[i]['max_score']
                    steps_check = steps_per_game[i] >= self.game_sample_freq
                    if score_check or steps_check:
                        score_counter.update([ np.round( np.average(all_scores[i][-self.window_average:]) ,1) ])
                        if score_check: 
                            num_score_check += 1
                        if steps_check:
                            num_step_check += 1
                        num_games_trained += 1

                        #sample new game
                        idx = np.random.randint(len(self.all_train_games))
                        if len(seen_games) < len(self.all_train_games):
                            while idx in seen_games:
                                idx = np.random.randint(len(self.all_train_games))

                        game = self.all_train_games[idx]
                        game_id = textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.env_step_limit)
                        new_env = gym.make(game_id)
                        self.train_envs[i] = new_env
                        self.current_train_ids[i] = idx
                        
                        seen_games.add(idx)
                        
                        if not self.debug_mode:
                            game_name = infos[i]['game'].metadata["uuid"].split("+")[-1]
                            wandb.log({  
                                "NumScoreCheck" : num_score_check,
                                "NumStepsCheck":  num_step_check,
                                "NumGamesTrained": num_games_trained,
                                "NumUniqueGamesTrained": len(seen_games), 
                                game_name +"_Score": np.average(all_scores[i][-self.window_average:]),
                                game_name +"_Steps": steps_per_episode[i]
                            },commit=False)

                        #reset this env
                        n_obs, n_infos = self.train_envs[i].reset()

                        next_desc_text[i] = clean_game_state("DESCRIPTION: "+ n_infos['description']) 
                        next_inventory_text[i] = clean_game_state("INVENTORY: "+ n_infos['inventory']) 
                        next_observation_text[i] = clean_game_state(n_obs) 
                        next_recipe_text[i] = clean_game_state((n_infos['extra.recipe'] or n_infos['extra.goal']))
                        next_prev_a_text[i] = 'None'
                        next_infos[i] = n_infos


                        all_scores[i] = []
                        steps_per_game[i] = 0
                        steps_per_episode[i] = 0
                        scores[i] = 0
                        dones[i] = False
                        last_scores[i] = 0

            desc_text = next_desc_text
            inventory_text = next_inventory_text
            observation_text = next_observation_text
            recipe_text = next_recipe_text
            prev_a_text = next_prev_a_text
            infos = next_infos

            #evaluate agent
            if not self.no_eval:
                if total_steps % self.eval_freq == 0:
                    self.agent.state_encoder.clone_hidden() 

                    total_normalized_scores = []
                    total_eval_steps = []
                    t_total_normalized_scores = []
                    t_total_eval_steps = []

                    for ep in tqdm(range(3),desc = 'Eval Trial'):
                        
                        eval_normalized_scores, eval_episode_steps = self.eval_episode('sampled',self.all_eval_envs)
                        total_normalized_scores.append(eval_normalized_scores)
                        total_eval_steps.append(eval_episode_steps)

                    avg_normalized_score_per_game = np.average(total_normalized_scores,axis=0) #20,
                    avg_steps_per_game = np.average(total_eval_steps,axis=0) #20,
                    avg_normalized_score_per_trial =  np.average(total_normalized_scores,axis=1) #10,
                    avg_steps_per_trial = np.average(total_eval_steps,axis=1) #10,

                    if not self.debug_mode:
                        obs, ifs = zip(*[env.reset() for env in self.eval_envs])

                        for i in range(len(eval_normalized_scores)):
                            game_name = ifs[i]['game'].metadata["uuid"].split("-")[-1]
                            wandb.log({

                                game_name +"_EvalNormalizedScore": eval_normalized_scores[i],
                                game_name +"_EvalSteps": eval_episode_steps[i]
                            }, commit=False)

                        wandb.log({
                            "EvalNormalizedAverageScores": np.average(avg_normalized_score_per_trial),
                            "EvalAverageSteps": np.average(avg_steps_per_trial)
                        })

                    with open("{}/{}_{}.json".format('json_files/',total_steps,self.experiment_name), 'w') as outfile:
                        json.dump(dict(score_counter), outfile)

                  
                    self.agent.state_encoder.init_hidden(self.batch_size)
                    self.agent.state_encoder.restore_hidden()
            
            total_steps += 1
            steps_per_game = [i + 1 for i in steps_per_game]

    def eval_episode(self,mode,envs):
        '''
        Run a single episode of all evaluation games and average score/steps across all games
        '''
        self.eval_envs = envs
        num_games = len(self.eval_envs)
        self.agent.state_encoder.init_hidden(num_games)
        

        scores          = [0] * num_games
        last_scores     = [0] * num_games
        running         = [1] * num_games
        episode_scores  = [0] * num_games
        episode_steps   = [0] * num_games
        steps_per_env   = [0] * num_games
        dones           = [False] * num_games
        
        obs, infos = zip(*[env.reset() for env in self.eval_envs])
        max_scores = [i['max_score'] for i in infos]

        desc_text,inventory_text,observation_text,recipe_text = get_game_text(obs,infos, num_games)
        prev_a_text = ["None" for i in range(num_games) ]

        total_steps = 0
        while True:

            gt_admissible  = get_filtered_admissible_commands(infos)

            with torch.no_grad():
                action_idxs, action_logits,action_mask, value = self.agent.act(desc_text,inventory_text,observation_text,prev_a_text,recipe_text,\
                                                                            scores=scores, predicted_valid_acts = predicted_valid_acts)
            print(steps_per_env)
            action_strs = [predicted_valid_acts[i][action_idxs[i].item()] for i in range(num_games)]

            #take step in game for each env
            next_obs, scores, dones , next_infos = zip(*[self.eval_envs[i].step(action_strs[i]) for i in range(num_games)])
            next_obs = [next_obs[i] for i in range(len(next_obs))]
            dones = [dones[i] for i in range(len(dones))]
            next_infos = list(next_infos)
            scores = [int(scores[i]) for i in range(len(scores))]
            reward = np.array(list(scores)) - np.array(last_scores)
            last_scores = np.array(list(scores))
            steps_per_env = [steps_per_env[i] + 1 for i in range(num_games)]

            next_desc_text,next_inventory_text,next_observation_text,next_recipe_text = get_game_text(next_obs,next_infos,num_games)
            next_prev_a_text = action_strs

            print("EvalStep: {}, Episode Step: {}\nScores: {} Rewards: {} Dones: {} Actions: {}".format( total_steps, steps_per_env, scores,reward, dones, action_strs))

            done_mask =  (~torch.tensor(dones)).float().to(self.device).unsqueeze(1)
            self.agent.state_encoder.reset_hidden(done_mask)

            for i, done in enumerate(dones):
                if done and running[i] == 1:
                    running[i] = 0
                    episode_steps[i] = steps_per_env[i]

            desc_text = next_desc_text
            inventory_text = next_inventory_text
            observation_text = next_observation_text
            recipe_text = next_recipe_text
            prev_a_text = next_prev_a_text
            infos = next_infos
            
            total_steps += 1

            if sum(running) == 0:
                break

        #fill envs that didnt finish within timeframe
        for i in range(len(running)):
            if running[i]:
                episode_steps[i] = steps_per_env[i]

        #average across all envs
        average_scores = np.average(scores)
        average_steps  = np.average(episode_steps)
        normalized_scores = scores  / np.array(max_scores)
        average_normalized_scores = np.average(normalized_scores)

        return normalized_scores, episode_steps

    def generate_conditional_targets(self, input_admissible):
        """
        input_admissible: lists of lists of valid actions for a single env
        """

        batch = len(input_admissible)
        template_targets = torch.zeros(batch,self.template_size) #y_t|s
        o1_template_targets = torch.zeros(batch,self.template_size, self.object_size)  #y_o1|s,t
        o2_o1_template_targets = torch.zeros(batch,self.template_size,self.object_size, self.object_size) #y_o2|o1,s,t
        for i, acts in enumerate(input_admissible):
            valid_acts = self.convert_commands_to_lists(acts)

            assert 'NO_OBJECT' in list(self.object2id.keys())
            assert 'NOT_ADMISSIBLE' in list(self.object2id.keys())
            no_obj_id = self.object2id["NO_OBJECT"]
            not_admissible_obj_id = self.object2id["NOT_ADMISSIBLE"]

            #fill in objects from admissible commands
            for act in valid_acts:
            
                #act is [template, obj1, obj2]
                t = act[0]
                template_idx = self.template2id[t]
                template_targets[i][template_idx] = 1

                #check how many objects template has
                num_objs = len(act) - 1
                if num_objs == 0:
                    o1_template_targets[i][template_idx][no_obj_id] = 1 #this template does not require any objects
                    o2_o1_template_targets[i][template_idx][no_obj_id][no_obj_id] = 1
                    
                elif num_objs == 1:
                    obj_id = self.object2id[act[1]]
                    o1_template_targets[i][template_idx][obj_id] = 1
                    #o2_o1_template_targets[i][template_idx][obj_id][no_obj_id] = 1

                elif num_objs == 2:
                    obj1_id = self.object2id[act[1]]
                    obj2_id = self.object2id[act[2]]
                    o1_template_targets[i][template_idx][obj1_id] = 1
                    o2_o1_template_targets[i][template_idx][obj1_id][obj2_id] = 1

            #fill inadmissible commands
            valid_templates = [valid_acts[i][0] for i in range(len(valid_acts))]
            for t in self.template2id.keys():
                if t not in valid_templates:
                    template_idx = self.template2id[t]
                    o1_template_targets[i][template_idx][not_admissible_obj_id] = 1 # #this template is not admissible, set flags for object targets
                    #import pdb;pdb.set_trace()
                    o2_o1_template_targets[i][template_idx][not_admissible_obj_id][not_admissible_obj_id] = 1
              
    
        return template_targets, o1_template_targets, o2_o1_template_targets

    def convert_commands_to_lists(self,admissible_commands):
        '''
        input: [open fridge, close fridge, take onion from fridge . . . 
        output [[open OBJ, fridge, None], [close OBJ, fridge, None], [take OBJ from OBJ, onion, fridge] ...
        '''
        valid_acts = []

        for act in admissible_commands:
            ents = self.extract_entities(act)
            for ent in ents:
                if ent is not None:
                    act = act.replace(ent, "OBJ")
            cmd = [act] + ents 
            valid_acts.append(cmd)
        return valid_acts

    def extract_entities(self, input_command):
    
        """
        Extract entities in order from given input command

        Example:
            input 'cut apple with knife'
            output [apple, knife]

            input 'close fridge'
            output [fride]

            input: 'look'
            output []
        """

        #find which ents from all_ents are present in command
        all_ents = list(self.object2id.keys())
        starting_command = input_command
        #if input_command == 'put orange bell pepper on shelf':

        ents = []
        idxs = []

        #check combo of three words
        three_words = [" ".join(item) for item in itertools.combinations(input_command.split(" "), 3)]
        for combo in three_words:
            if combo in all_ents:
                ents.append(combo)
                input_command = input_command.replace(combo,"OBJ")
 
                
        two_words = [" ".join(item) for item in itertools.combinations(input_command.split(" "), 2)]
        for combo in two_words:
            if combo in all_ents:
                ents.append(combo)
                input_command = input_command.replace(combo,"OBJ")
           

        words = re.findall(r'\w+', input_command)
        for word in words:
            if word in all_ents:
                ents.append(word)
        

        if len(ents) == 0:
            return []
        elif len(ents) == 1:
            return [ents[0]]

        #if more than one ent, determine which ent goes to position 1 or position 2
        else:
            ent1 = starting_command.replace(ents[0], "OBJ")
            ent2 = starting_command.replace(ents[1], "OBJ")
            ent1_pos = ent1.find("OBJ")
            ent2_pos = ent2.find("OBJ")
            if ent1_pos < ent2_pos:
                return [ents[0], ents[1]]
            elif ent1_pos > ent2_pos:
                return [ents[1], ents[0]]

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

    def train_backplay(self):
        '''
        Use walkthrough to train starting from the end and slowly progress backwards
        '''
        episodes               = [0] * self.batch_size
        steps_per_episode      = [0] * self.batch_size
        scores                 = [0] * self.batch_size
        last_scores            = [0] * self.batch_size
        current_training_steps = [0] * self.batch_size
        steps_per_walkthrough  = [0] * self.batch_size
        dones                  = [False] * self.batch_size
        last_x_episode_score =  [0] * self.batch_size
        last_x_episode_steps = [self.env_step_limit] * self.batch_size

        all_scores                  = [[] for i in range(self.batch_size)]
        average_scores              = [[] for i in range(self.batch_size)]
        all_steps_per_episode       = [[] for i in range(self.batch_size)]
        average_steps_per_episode   = [[] for i in range(self.batch_size)]

        obs, infos = zip(*[env.reset() for env in self.train_envs])
        walkthroughs = [info['extra.walkthrough'] for info in infos]

        #since im giving the agent inventory and recipe as input
        for i in range(self.batch_size):
            walkthroughs[i].remove('inventory')
            walkthroughs[i].remove('examine cookbook')

        t = [len(walkthrough)-1 for walkthrough in walkthroughs]
        
        #walk agent to t
        envs, next_obs, bert_text, feedback_text, scores, dones, infos = zip(*[self.walk_env_to_t(self.train_envs[i],t[i],walkthroughs[i]) for i in range(self.batch_size)])
        envs = list(envs)
        bert_text = list(bert_text)
        feedback_text = list(feedback_text)
        scores = list(scores)
        dones = list(dones) 
        infos = list(infos)
                              
        a2c_transitions = []
        converged = False
        average_batched_scores = 0
        average_batched_steps  = self.env_step_limit
        total_steps = 0
        #self.agent.bert.eval()
        while total_steps <= self.training_steps:
           
            gt_admissible = [ infos[i]['admissible_commands'] for i in range(self.batch_size) ]
            for acts in gt_admissible:
                for ac in acts[:]:
                    if (ac.startswith('examine') and ac != 'examine cookbook') or ac == 'look' or ac == 'inventory':
                        acts.remove(ac)

           
            bert_encoding = self.agent.encode_bert(bert_text)
           
            #generate actions
            if self.use_gt_commands:
                predicted_valid_acts = gt_admissible
            else:
                with torch.no_grad():
                    logits_t, logits_o1, logits_o2 = self.command_generator(bert_encoding)
                predicted_valid_acts = self.get_generated_valid_actions(logits_t, logits_o1, logits_o2)
                template_targets, o1_targets, o2_targets = self.generate_conditional_targets(gt_admissible)
                trans = logits_t, logits_o1, logits_o2, template_targets, o1_targets, o2_targets
                metrics = self.calc_f_score(trans)

            action_idxs, action_logits,action_mask, value = self.agent.act(bert_encoding, feedback_text, scores, predicted_valid_acts)   

            action_strs = [predicted_valid_acts[i][action_idxs[i].item()] for i in range(self.batch_size)]
            #action_strs = [ infos[i]['admissible_commands'][np.random.randint(len(infos[i]['admissible_commands']))] for i in range(self.batch_size) ]

            #take step in game for each env
            next_obs, scores, dones , next_infos = zip(*[envs[i].step(action_strs[i]) for i in range(self.batch_size)])                
            next_obs = [next_obs[i] for i in range(len(next_obs))]
            dones = [dones[i] for i in range(len(dones))]
            next_infos = list(next_infos)
            scores = [int(scores[i]) for i in range(len(scores))]
            reward = np.array(list(scores)) - np.array(last_scores)
            last_scores = np.array(list(scores))
            steps_per_episode = [steps_per_episode[i] + 1 for i in range(self.batch_size)]
            if self.use_extra_reward:
                for i in range(self.batch_size):
                    if next_infos[i]['won']:
                        reward[i] = reward[i]#10
                    elif next_infos[i]['lost']:
                        reward[i] = -0.1#-10


            sys.stdout.flush()
            print("Step: {}, Episode Step: {}\nScores: {} Rewards: {} Dones: {} Actions: {}".format(total_steps, steps_per_episode[5], scores[5],reward[5], dones[5], action_strs[5]))
            if not self.use_gt_commands:
                print("Extra:")
                for i in range(self.batch_size):
                    print("\t", i , set(predicted_valid_acts[i]) - set(gt_admissible[i]))
                print("Missed:")
                for i in range(self.batch_size):
                    print("\t", i ,  set(gt_admissible[i]) - set(predicted_valid_acts[i]))

                print("Predicted: {}\nGT: {}\n".format(predicted_valid_acts[5],gt_admissible[5]))
                print("Template FPR: {}\nO1 FPR: {}\nO2 FPR: {}".format(metrics['template'],metrics['o1'], metrics['o2']))
                print("Extra: ", [len([item for item in predicted_valid_acts[i] if item not in gt_admissible[i]]) for i in range(self.batch_size) ])
                print("Missed: ",[len([item for item in gt_admissible[i] if item not in predicted_valid_acts[i]]) for i in range(self.batch_size) ])

            named_action_scoring_probs = [(s, p.item()) for s,p in zip(predicted_valid_acts[5], self.agent.actor.masked_softmax(action_logits,action_mask,dim=1)[5])]
            for i in range(len(named_action_scoring_probs)):
                print(named_action_scoring_probs[i])
            print()

            next_bert_text =        [self.clean_game_state("DESCRIPTION: "+ next_infos[i]['description'] + " INVENTORY: "+ next_infos[i]['inventory']) for i in range(self.batch_size)]
            next_feedback_text =    [self.clean_game_state((next_infos[i]['extra.recipe'] or next_infos[i]['extra.goal']))+ "|" + self.clean_game_state(next_obs[i]) + "|" + action_strs[i] for i in range(self.batch_size) ]

            done_mask =  (~torch.tensor(dones)).float().to(self.device).unsqueeze(1)
            rw = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            self.agent.state_encoder.reset_hidden(done_mask)

            a2c_transitions.append((
                action_idxs,
                action_logits,
                action_mask,
                value,
                rw,
                done_mask
            ))

            #update agent
            if len(a2c_transitions) >= self.bptt:
                self.agent.state_encoder.clone_hidden()
                next_bert_encoding = self.agent.encode_bert(next_bert_text)
                next_value = self.agent.act(next_bert_encoding, next_feedback_text, scores, None )
                returns, advantages = self.discount_reward(a2c_transitions, next_value)

                if not self.debug_mode:
                    wandb.log({
                        'Advantage': advantages[-1].median().item(),
                        'Value': value.mean().item()
                    },commit=False)

                self.update_a2c(a2c_transitions, returns, advantages)
                del a2c_transitions[:]
                self.agent.state_encoder.restore_hidden() 

            #check if any env is done and reset
            for i, done in enumerate(dones):
                if done:
                    #keep track of stats
                    score = scores[i]
                    step = steps_per_episode[i]
                    all_scores[i].append(score)
                    all_steps_per_episode[i].append(step)
                    
                    episodes[i] += 1 
                    sys.stdout.flush()
                    print('-' * 10, 'New Episode', '-'*10)
                    print('Batch ID: ', i, 'Episode Score: ', score, 'Episode Steps: ', step)

                    #reset this env
                    env, n_obs, n_bert, n_feedback, s, d, n_info = self.walk_env_to_t(envs[i],t[i],walkthroughs[i])

                    envs[i] = env           
                    next_feedback_text[i] = n_feedback
                    next_bert_text[i] = n_bert
                    next_obs[i] = n_obs
                    next_infos[i] = n_info

                    steps_per_episode[i] = 0
                    scores[i] = s
                    dones[i] = d
                    last_scores[i] = s

            #check if time to decrease walkthrough or sample new game
            for i in range(self.batch_size):
                score_check = len(all_scores[i]) >= self.window_average and np.average(all_scores[i][-self.window_average:]) == infos[i]['max_score']
                steps_check = steps_per_walkthrough[i] >= self.max_walkthrough_steps
                if score_check or steps_check:
                    #decrease walkthrough by one
                    t[i] = t[i] - 1

                    if t[i] < 0:
                        #sample one new env 
                        idx = np.random.randint(len(self.all_train_games))
                        game = self.all_train_games[idx]
                        game_id = textworld.gym.register_game(game, request_infos=self.env_options, max_episode_steps= self.env_step_limit)
                        new_env = gym.make(game_id)
                        envs[i] = new_env

                        #reset walkthrough
                        _, info = envs[i].reset()
                        walkthroughs[i] = info['extra.walkthrough']
                        walkthroughs[i].remove('inventory')
                        walkthroughs[i].remove('examine cookbook')
                        t[i] = len(walkthroughs[i]) - 1

                        if not self.debug_mode:
                            game_name = infos[i]['game'].metadata["uuid"].split("-")[-1]
                            wandb.log({     
                                game_name +"_Score": np.average(all_scores[i][-self.window_average:])
                                #game_name +"_Steps": steps_per_episode[i]
                            },commit=False)
                    
                    #reset this env
                    env, n_obs, n_bert, n_feedback, s, d, n_info = self.walk_env_to_t(envs[i],t[i],walkthroughs[i])

                    envs[i] = env           
                    next_feedback_text[i] = n_feedback
                    next_bert_text[i] = n_bert
                    next_obs[i] = n_obs
                    next_infos[i] = n_info

                    steps_per_episode[i] = 0
                    scores[i] = s
                    dones[i] = d
                    last_scores[i] = s 
                    all_scores[i] = []
                    steps_per_walkthrough[i] = 0

            feedback_text = next_feedback_text
            bert_text = next_bert_text
            infos = next_infos
            obs = next_obs

            #evaluate agent
            if total_steps % self.eval_freq == 0:
                print("Running Eval . . .")
                self.agent.state_encoder.clone_hidden() 
                average_eval_scores, average_normalized_scores, eval_scores,eval_normalized_scores, average_eval_steps, eval_episode_steps = self.eval()

                if not self.debug_mode:
                    obs, infos = zip(*[env.reset() for env in self.eval_envs])
                    
                    for i in range(len(eval_scores)):
                        game_name = infos[i]['game'].metadata["uuid"].split("-")[-1]
                        wandb.log({
                            game_name +"_EvalScore": eval_scores[i],
                            # game_name +"_EvalNormalizedScore": eval_normalized_scores[i],
                            # game_name +"_EvalSteps": eval_episode_steps[i]
                        }, commit=False)

                    wandb.log({
                        "EvalNormalizedAverageScores": average_normalized_scores,
                        "EvalAverageSteps": average_eval_steps
                    })

                    with open("{}.json".format(self.experiment_name), 'w') as outfile:
                        json.dump(dict(coun), outfile)

                self.agent.state_encoder.init_hidden(self.batch_size)
                self.agent.state_encoder.restore_hidden()

            total_steps += 1
            steps_per_walkthrough = [i + 1 for i in steps_per_walkthrough]
