
from torch.utils.data import Dataset
import json
import numpy as np
from transformers import BertTokenizer
import torch
import itertools

import re

class AdmissibleCommandsClassificationDataset(Dataset):
    def __init__(self, data_file, template2id, object2id, max_seq_length, bert_model_type):
        with open(data_file) as json_file:
            self.data = json.load(json_file)

        self.template2id = template2id
        self.object2id = object2id

        self.template_size = len(template2id)
        self.object_size = len(object2id)

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_type, do_lower_case=False)

        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        datapoint = self.data[idx]
        input_admissible = datapoint['admissible_commands']
        state = datapoint['state']
        
        #if self.encoder_type == 'bert':
        
        x = '[CLS] ' + state  + ' [SEP]'
        x = self.tokenizer.tokenize(x)
        input_ids = self.tokenizer.convert_tokens_to_ids(x)
        
        input_mask = [1] * len(input_ids) 
        diff = (self.max_seq_length - len(input_ids))
        if diff > 0:
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
        else:
            input_ids = input_ids[:self.max_seq_length]
            input_mask = input_mask[:self.max_seq_length]
        
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0)
        state = torch.cat([input_ids, input_mask],dim=0)
    
        # elif self.encoder_type == 'rnn':
        #     state = self.prepare_state_rnn(state)
            

        template_targets = torch.zeros(self.template_size) #y_t|s
        o1_template_targets = torch.zeros(self.template_size, self.object_size)  #y_o1|s,t
        o2_o1_template_targets = torch.zeros(self.template_size,self.object_size, self.object_size) #y_o2|o1,s,t

        valid_acts = self.convert_commands_to_lists(input_admissible)

        assert 'NO_OBJECT' in list(self.object2id.keys())
        assert 'NOT_ADMISSIBLE' in list(self.object2id.keys())
        no_obj_id = self.object2id["NO_OBJECT"]
        not_admissible_obj_id = self.object2id["NOT_ADMISSIBLE"]

        #fill in objects from admissible commands
        for act in valid_acts:
         
            #act is [template, obj1, obj2]
            t = act[0]
            template_idx = self.template2id[t]
            template_targets[template_idx] = 1

            #check how many objects template has
            num_objs = len(act) - 1
            if num_objs == 0:
                #continue
                o1_template_targets[template_idx][no_obj_id] = 1 #this template does not require any objects
                o2_o1_template_targets[template_idx][no_obj_id][no_obj_id] = 1
               
            elif num_objs == 1:
                obj_id = self.object2id[act[1]]
                o1_template_targets[template_idx][obj_id] = 1
                o2_o1_template_targets[template_idx][obj_id][no_obj_id] = 1

            elif num_objs == 2:
                obj1_id = self.object2id[act[1]]
                obj2_id = self.object2id[act[2]]
                o1_template_targets[template_idx][obj1_id] = 1
                o2_o1_template_targets[template_idx][obj1_id][obj2_id] = 1

        #fill inadmissible commands
        valid_templates = [valid_acts[i][0] for i in range(len(valid_acts))]
        for t in self.template2id.keys():
            if t not in valid_templates:
                template_idx = self.template2id[t]
                o1_template_targets[template_idx][not_admissible_obj_id] = 1 # #this template is not admissible, set flags for object targets
                #import pdb;pdb.set_trace()
                o2_o1_template_targets[template_idx][not_admissible_obj_id][not_admissible_obj_id] = 1
                       
        return torch.LongTensor(state), template_targets, o1_template_targets, o2_o1_template_targets

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

    def prepare_state_rnn(self,state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            state_description = state_description.replace(rm, '')
       
        state_description = state_description.split('|')

        ret = [self.sp.encode_as_ids('<s>' + s_desc + '</s>') for s_desc in state_description]

        return self.pad_sequences(ret, maxlen=self.max_seq_length)

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
