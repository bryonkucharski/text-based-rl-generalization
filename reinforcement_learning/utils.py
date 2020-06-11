import os
import glob
import logger
import re

import sentencepiece

import torch.nn.functional as F
import torch
import numpy as np 
import random

def clean_game_state(state):
    lines = state.split("\n")
    cur = [a.strip() for a in lines]
    cur = ' '.join(cur).strip().replace('\n', '').replace('---------', '')
    cur = re.sub("(?<=-\=).*?(?=\=-)", '', cur)
    cur = re.sub('[$_\\|/>]', '', cur)
    cur = cur.replace("-==-", '').strip()
    cur = cur.replace("\\", "").strip()
    return cur   

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def calc_f_score(trans):
            
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

    t_recall, t_precision, t_fscore,_ = precision_recall_fscore_support(template_targets,t_probs,average = 'weighted')

    N = self.batch_size
    for b in range(N):
        o1_p, o1_r, o1_f,_  = precision_recall_fscore_support(o1_targets[b],o1_probs[b],average = 'weighted')
        o1_fscore += o1_f
        o1_precision += o1_p
        o1_recall += o1_r

        # import random;
        # if random.random() < 0.0005:
        #     import pdb;pdb.set_trace()

        #average o2 f score over templates and batches
        o2_p, o2_r, o2_f = 0,0,0
        for t in range(len(self.template2id)):

            temp_p, temp_r, temp_f,_  = precision_recall_fscore_support(o2_targets[b][t],o2_probs[b][t],average = 'weighted')
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

def configure_logger(log_dir,experiment_name):
    logger.configure(log_dir + "/"+experiment_name, format_strs=['log'])
    return logger.log

def discount_reward(transitions, last_values, gamma):
    returns, advantages = [], []
    R = last_values.data
    for t in reversed(range(len(transitions))):
        _, _,_, values, rewards, done_masks = transitions[t]
        R = rewards + gamma * R * done_masks
        adv = R - values
        returns.append(R)
        advantages.append(adv)
    return returns[::-1], advantages[::-1]

def update_a2c(a2c_transitions, returns, advantages, policy_coeff = 1.0,value_coeff = 0.5, entropy_coeff = 0.01):
    assert len(a2c_transitions) == len(returns) == len(advantages)
    loss = 0
    for i, (trans, ret, adv) in enumerate(zip(a2c_transitions, returns, advantages)):
        action_ids, action_logits, action_mask, value, rw, done_mask = trans
        
        # Policy Gradient Loss
        adv                 = adv.detach() 

        #masking is needed because commands are padded to all be equal length. 
        all_probs           = masked_softmax(action_logits,action_mask,dim=1) 
        all_log_probs       = masked_log_softmax(action_logits,action_mask,dim=1)

        action_log_probs    = all_log_probs.gather(1, action_ids)
        policy_loss         = policy_coeff  * (-action_log_probs * adv).mean()
        value_loss          = value_coeff   * F.mse_loss(value, ret)
        entropy_loss        = entropy_coeff * (-all_probs* all_log_probs).mean()

        loss +=  (value_loss  + policy_loss - entropy_loss)

    return loss, policy_loss, value_loss, entropy_loss

def masked_softmax(vec, mask, dim=1, mask_fill_value=-1e32):
    '''
    Taken from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L231
    '''
    mask = mask.float()
    masked_vector = vec.masked_fill((1 - mask).bool(), mask_fill_value)
    return F.softmax(masked_vector, dim=dim)

def masked_log_softmax(vec,mask,dim,epsilon = 1e-45):

    '''
    Taken from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
    '''
    mask = mask.float()
    vec = vec + (mask + 1e-45).log()
    return F.log_softmax(vec, dim=dim)

def step_multi_env(envs,action_strs,last_scores):
    batch_size = len(envs)
    next_obs, scores, dones , next_infos = zip(*[envs[i].step(action_strs[i]) for i in range(batch_size)])
    next_obs = [next_obs[i] for i in range(len(next_obs))]
    dones = [dones[i] for i in range(len(dones))]
    next_infos = list(next_infos)
    scores = [int(scores[i]) for i in range(len(scores))]
    reward = np.array(list(scores)) - np.array(last_scores)
    next_scores = np.array(list(scores))

    return next_obs, scores, dones , next_infos, reward, next_scores

def get_filtered_admissible_commands(infos):
    batch_size = len(infos)
    gt_admissible = [ infos[i]['admissible_commands'] for i in range(batch_size) ]
    for acts in gt_admissible:
        for ac in acts[:]:
            if (ac.startswith('examine') and ac != 'examine cookbook') or ac == 'look' or ac == 'inventory':
                acts.remove(ac)
    return gt_admissible

def get_game_text(obs,infos, size):
    desc_text = [clean_game_state("DESCRIPTION: "+ infos[i]['description']) for i in range(size) ]
    inventory_text = [clean_game_state("INVENTORY: "+ infos[i]['inventory']) for i in range(size) ]
    observation_text = [clean_game_state(obs[i]) for i in range(size) ]
    recipe_text = [clean_game_state((infos[i]['extra.recipe'] or infos[i]['extra.goal'])) for i in range(size) ]
    return desc_text,inventory_text,observation_text,recipe_text

def get_train_games(data_dir, training_size = 20, difficulty_level=4):
    game_file_names = []
    game_path = data_dir + "/train_" + str(training_size) + "/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)
    return game_file_names

def get_eval_games(data_dir, training_size = 20, difficulty_level=4):
    game_file_names = []
    game_path = data_dir + "/valid/difficulty_level_" + str(difficulty_level)
    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
    else:
        game_file_names.append(game_path)
    return game_file_names

def get_gt_recipe(infos,object2id,direction2id):
        
    #will only work for diff3?
    ings = infos['game'].metadata['ingredients']
    recipe = []
    for ing in ings:
        ing_gt = [0] * len(object2id)
        direction_gt = [0] * len(direction2id)
        ing_gt[object2id[ing[0]]] = 1
        direction_gt[direction2id[ing[1]]] = 1
        direction_gt[direction2id[ing[2]]] = 1
        recipe_encoding = ing_gt + direction_gt
        recipe += recipe_encoding

    return recipe

def get_generated_valid_actions(logits_t, logits_o1, logits_o2, id2template,id2object):

    batch_size = logits_t.size()[0]

    t_preds = torch.ge(torch.sigmoid(logits_t), 0.5).float()
    o1_preds  =  torch.ge(torch.sigmoid(logits_o1), 0.5).float()
    o2_preds =  torch.ge(torch.sigmoid(logits_o2), 0.5).float()

    commands = [[] for i in range(batch_size)]

    for i in range(batch_size):
        t_p = t_preds[i]
        template_idxs = (t_p == 1).nonzero()
        
        for idx in template_idxs:
            template = id2template[idx.item()]
            num_obj = template.count('OBJ')

            if num_obj == 0:
                commands[i].append(template)

            elif num_obj == 1:
                o1_p = o1_preds[i][idx].squeeze()
                o1_idxs = (o1_p == 1).nonzero()
                for idx in o1_idxs:
                    if id2object[idx.item()] != 'NOT_ADMISSIBLE' and id2object[idx.item()] != 'NO_OBJECT':
                        command = template.replace("OBJ",id2object[idx.item()])
                        commands[i].append(command)
            elif num_obj == 2:
                
                o2_p = o2_preds[i][idx].squeeze() #2d tensor
                o2_idxs = (o2_p == 1).nonzero()
                for idx in o2_idxs:
                    obj1 = id2object[idx[0].item()]
                    obj2 = id2object[idx[1].item()]
                    if obj1 != 'NOT_ADMISSIBLE' and obj1 != 'NO_OBJECT' and obj2 != 'NOT_ADMISSIBLE' and obj2 != 'NO_OBJECT':
                        command = template.replace("OBJ",obj1,1).replace("OBJ",obj2,2)
                        commands[i].append(command)
                

    return commands


class _SentencepieceProcessor:
    '''
    See issue https://github.com/dmlc/gluon-nlp/issues/1233
    '''
    def __init__(self, path):
        self._vocab_file = path
        self._processor = sentencepiece.SentencePieceProcessor()
        self._processor.Load(path)

    def __len__(self):
        return len(self._processor)

    def encode_as_ids(self,input):
        return self._processor.encode_as_ids(input)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_processor"] = None
        state["_vocab_file"] = None
        return state, self._vocab_file

    def __setstate__(self, d):
        self.__dict__, self.vocab_file = d
        self._processor = sentencepiece.SentencePieceProcessor()
        self._processor.Load(self.vocab_file)