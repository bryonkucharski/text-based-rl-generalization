
import os
import argparse

from trainers import AdmissibleCommandTrainer

def parse_args():
    parser = argparse.ArgumentParser()
            
    parser.add_argument('--lr', default = 2e-5,type = float)
    parser.add_argument('--template_coeff', default = 1.0,type = float)
    parser.add_argument('--object1_coeff', default = 5,type = float)
    parser.add_argument('--object2_coeff', default = 10,type = float)
    parser.add_argument('--weight_decay', default = 1e-4,type = float)
    parser.add_argument('--adam_epsilon', default = 1e-8,type = float)

    parser.add_argument('--batch_size',  default = 8,    type = int)
    parser.add_argument('--embedding_size', default = 50,  type = int)
    parser.add_argument('--max_seq_len',  default = 350,  type = int)
    parser.add_argument('--print_freq', default = 1000, type = int)
    parser.add_argument('--bert_hidden_size', default = 768,  type = int)
    parser.add_argument('--bert_transformation_size', default = 300,  type = int)
    parser.add_argument('--gradient_accumulation_steps',default = 1,  type = int)
    parser.add_argument('--max_grad_norm',default = 1,  type = int)
    parser.add_argument('--warmup_steps',default = 0,  type = int)
    parser.add_argument('--start_epoch',default = 0,  type = int)
    parser.add_argument('--end_epoch',default = 20,  type = int)

    parser.add_argument('--train', default = False, action='store_true')
    parser.add_argument('--verbose', default = False, action='store_true')
    parser.add_argument('--load_checkpoint', default = False, action='store_true')

    parser.add_argument('--project_name', default = 'AC_Classifier',help ='name of project for wandb logging')
    parser.add_argument('--experiment_name', default = 'bert_full_data_compressed')
    parser.add_argument('--data_file', default = 'data/data.json')
    parser.add_argument('--object_file', default = 'data/cooking_games_entities.txt')
    parser.add_argument('--template_file', default = 'data/template2id.txt')
    parser.add_argument('--log_dir', default = 'logs/')
    parser.add_argument('--checkpoint_dir', default = '/checkpoint/')
    parser.add_argument('--bert_model_type', default = 'bert-base-cased')

    return parser.parse_args()

if __name__ == "__main__":
    params = parse_args() 

    if params.train:
        trainer = AdmissibleCommandTrainer(params)
        trainer.train()