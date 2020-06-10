import re
import os 
import torch 


def save_pickle(path,name, model):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path +"/" + name + ".pkl", "wb") as f:
        torch.save(model.state_dict(), f)

def load_pickle(path,model):

    with open(path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    return model

def get_train_valid_test_split(dataset_size, split):
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices

def get_eval_metrics(entity_1_logits,entity_2_logits,relation_logits,ent1_y,ent2_y,rel_y,entity_thres, relation_thres):
    """
        Get Precision, Recall, F Score of each classifier
    """
    sigmoid = nn.Sigmoid()
    entity_1_probs = sigmoid(entity_1_logits)
    entity_2_probs = sigmoid(entity_2_logits)
    rel_probs = sigmoid(relation_logits)

    entity_1_probs[entity_1_probs > entity_thres] = 1
    entity_1_probs[entity_1_probs <= entity_thres] = 0
    entity_2_probs[entity_2_probs > entity_thres] = 1
    entity_2_probs[entity_2_probs <= entity_thres] = 0
    rel_probs[rel_probs > relation_thres] = 1
    rel_probs[rel_probs <= relation_thres] = 0
    
    entity_1_probs = entity_1_probs.detach().cpu().numpy()
    entity_2_probs = entity_2_probs.detach().cpu().numpy()
    rel_probs = rel_probs.detach().cpu().numpy()
    ent1_y = ent1_y.to('cpu').numpy()
    ent2_y = ent2_y.to('cpu').numpy()
    #rel_indicies = rel_indicies.detach().cpu().numpy()
    rel_y = rel_y.to('cpu').numpy()

    e1_precision, e2_precision, rel_precision = 0, 0, 0
    e1_recall, e2_recall, rel_recall = 0, 0, 0
    e1_fscore, e2_fscore, rel_fscore = 0, 0, 0
    for i in range(rel_y.shape[0]):
        e1_p, e1_r, e1_f, _ = precision_recall_fscore_support(ent1_y[i],entity_1_probs[i],average = 'weighted')
        e2_p, e2_r, e2_f, _ = precision_recall_fscore_support(ent2_y[i],entity_2_probs[i],average='weighted')
        rel_p, rel_r, rel_f, _ = precision_recall_fscore_support(rel_y[i],rel_probs[i],average = 'weighted')
        e1_precision += e1_p
        e2_precision += e2_p
        rel_precision += rel_p
       
        e1_recall += e1_r
        e2_recall += e2_r
        rel_recall += rel_r

        e1_fscore += e1_f
        e2_fscore += e2_f
        rel_fscore += rel_f
 
    N = rel_y.shape[0]
    metrics = {
        'e1': (  e1_precision/N, e1_recall/N, e1_fscore/N ),
        'e2': (  e2_precision/N, e2_recall/N, e2_fscore/N ),
        'rel': ( rel_precision/N, rel_recall/N,rel_fscore/N )    
    }
    return metrics

