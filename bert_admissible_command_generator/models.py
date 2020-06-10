import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AdmissibleCommandGenerator(nn.Module):
    def __init__(self, num_templates,num_objects,embedding_size,state_hidden_size, state_transformed_size=None):
        super(AdmissibleCommandGenerator, self).__init__()

        self.embedding_size = embedding_size

        self.num_templates = num_templates
        self.num_objects = num_objects
      
        template_input_size     = state_hidden_size + self.embedding_size
        o1_input_size           = state_hidden_size + 2*self.embedding_size
        o2_o1_input_size        = state_hidden_size + 3*self.embedding_size

        self.template_embeddings = nn.Embedding(num_templates ,self.embedding_size)
        self.o1_embeddings       = nn.Embedding(num_objects   ,self.embedding_size)
        self.o2_embeddings       = nn.Embedding(num_objects   ,self.embedding_size)
        
        self.template_classifier = nn.Sequential(
                            nn.Linear(template_input_size, int(template_input_size/2)),
                            nn.ReLU(),
                            nn.Linear(int(template_input_size/2), 1)         
        )
        self.o1_template_classifier = nn.Sequential( 
                            nn.Linear(o1_input_size ,int(o1_input_size/2)),
                            nn.ReLU(),
                            nn.Linear(int(o1_input_size/2), 1)
        )

        self.o2_o1_template_classifier = nn.Sequential( 
                            nn.Linear(o2_o1_input_size ,int(o2_o1_input_size/2)),
                            nn.ReLU(),
                            nn.Linear(int(o2_o1_input_size/2), 1)
        )

        #self.bert_transformation = nn.Linear(state_hidden_size, state_transformed_size)

    def forward(self, encoded_state):
        
        #Goal: generate P(admissible_actions | state)
        #       P(admissible_actions | state) = P(templates,o1,o2|state)
        #       P(templates,o1,o2|state) = P(templates |state) * P(o1|templates,state) * P(o2 | o1, templates,state)


        #downsample BERT to use much less RAM
        #encoded_state = self.bert_transformation(encoded_state) 

        batch_size = encoded_state.shape[0] #batch x state_hidden_dim 
       
        #P(t  | s)
        template_embed = self.template_embeddings.weight                            # num_templates x embedding_size    
        template_embed = template_embed.unsqueeze(0).expand(batch_size,-1,-1)       #batch x num_templates x embedding_size

        encoded_state = encoded_state.unsqueeze(1).expand(-1,self.num_templates,-1) #batch x num_templates x hidden_size

        template_state_input = torch.cat([encoded_state,template_embed],dim=-1)     #batch x num_templates x 2*embedding_size
        template_logits = self.template_classifier(template_state_input)            #batch x num_templates x 1 
        template_logits = template_logits.squeeze(-1)                               #batch x num_templates 
        
        #P(o1 | s,t)
        
        object1_embed = self.o1_embeddings.weight # num_objects x embedding_size
        object1_embed = object1_embed.unsqueeze(0).unsqueeze(1).expand(batch_size,self.num_templates,-1,-1) # batch x num_templates x num_objects x embedding_size
        
        template_embed = template_embed.unsqueeze(2).expand(-1,-1,self.num_objects,-1)                      # batch x num_templates x num_objects x embedding_size
        encoded_state = encoded_state.unsqueeze(2).expand(-1,-1,self.num_objects, -1)                       # batch x num_templates x num_objects x embedding_size

        o1_template_state_input = torch.cat([encoded_state,template_embed, object1_embed],dim=-1)           # batch x num_templates x num_objects x 3*embedding_size
        o1_template_logits = self.o1_template_classifier(o1_template_state_input)                           # batch x num_templates x num_objects x 1
        o1_template_logits = o1_template_logits.squeeze(-1) # batch x num_templates x num_objects


        #P(o2 | s,t,o1)
        encoded_state = encoded_state.unsqueeze(2).expand(-1,-1,self.num_objects, -1,-1)    # batch x num_templates x num_object x num_objects x hidden
        template_embed = template_embed.unsqueeze(2).expand(-1,-1,self.num_objects,-1, -1)  # batch x num_templates x num_object x num_objects x embedding_size

        object1_embed = self.o1_embeddings.weight #num_objects x embedding_size
        object2_embed = self.o2_embeddings.weight #num_objects x embedding_size

        #Note: order matters

        object1_embed = object1_embed.unsqueeze(0).expand(self.num_objects, -1, -1) # num_objects x num_objects x embedding_size
        object2_embed = object2_embed.unsqueeze(1).expand(-1,self.num_objects, -1)  # num_objects x num_objects x embedding_size

        o2_o1_embed = torch.cat([object1_embed,object2_embed],dim=-1)                                       # num_objects x num_objects x 2 * embedding_size 
        o2_o1_embed = o2_o1_embed.unsqueeze(0).unsqueeze(0).expand(batch_size,self.num_templates,-1,-1,-1)  # batch x num_templates x num_object x num_objects x 2*embedding_size
        o2_o1_template_state_input = torch.cat([encoded_state,template_embed,o2_o1_embed],dim=-1)           # batch x num_templates x num_object x num_objects x 4*embedding_size

        o2_o1_template_logits = self.o2_o1_template_classifier(o2_o1_template_state_input)  # batch x num_templates x num_object x num_objects x 1
        o2_o1_template_logits = o2_o1_template_logits.squeeze(-1)                           # batch x num_templates x num_object x num_objects

        return template_logits, o1_template_logits, o2_o1_template_logits