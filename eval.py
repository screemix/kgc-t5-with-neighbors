import math
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from pymongo import MongoClient
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, T5ForConditionalGeneration, T5Config, PretrainedConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import pandas as pd
import pickle
from typing import Dict
from collections import defaultdict
import json
from loguru import logger

from transformers import AutoTokenizer
import argparse, sys

from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name, get_optimizer # noqa: E402

parser = argparse.ArgumentParser()

parser.add_argument("--from_hub", help="Name of the repository and the model from huggingface model hub, e.g. DeepPavlov/t5-wikidata5M-with-neighbors", default=None)
parser.add_argument("--cpt_path", help="Path to the model's weigths", default=None)
parser.add_argument("--model_cfg", help="Name of model config, e.g. t5-small", default='t5-small')
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer', default='t5-small')
parser.add_argument("--gpu", help="id of gpu using for evaluation", type=int)
parser.add_argument("--transductive", help="0 for inductive datasets, 1 for transductive", type=bool)
parser.add_argument("--neighborhood", help="1 to use neighborhood in the input, 0 otherwise", type=bool)
parser.add_argument("--filter_unknown_entities", help="Whether to filter generated enities that are not among known entities, applicable only to transductive datasets", type=bool)
parser.add_argument("--filter_known_links", help="Whether to filter results that already persist in train or valid dataset", type=bool)
parser.add_argument("--mongodb_port", help="Port of the mongodb collection with the dataset", default=27017, type=int)
parser.add_argument("--entity_mapping_path", help="Path to the entity2text mapping", default="data/mappings/wd5m_aliases_entities_v3.txt")
parser.add_argument("--relation_mapping_path", help="Path to the relation2text mapping", default="data/mappings/wd5m_aliases_relations_v3.txt")
parser.add_argument("--input_db", help="Name of the mongo database that stores wikidata5m dataset", default='wikidata5m')
parser.add_argument("--verbalized_eval_collection", help="Name of the collection that stores verbalized KG for evaluation", default='verbalized_test')
parser.add_argument("--train_collection_input", help="Name of the collection that stores train KG", default='train-set')
parser.add_argument("--valid_collection_input", help="Name of the collection that stores valid KG", default='valid-set')
parser.add_argument("--test_collection_input", help="Name of the collection that stores test KG", default='test-set')
parser.add_argument("--output_file", help="File to output scores", default='scores/output_scores.txt')

args = parser.parse_args()


class EvalBatch:
    def __init__(self, items, tokenizer, max_length):
        self.inputs = [item['input'] for item in items]
        self.target_text = [item["outputs"] for item in items]
        self.relations = [item['relation'] for item in items]
        self.heads = [item['head'] for item in items]
        
        encode_plus_kwargs = {'truncation': True, 'padding': 'longest', 'pad_to_multiple_of': 1}
        self.inputs_tokenized = tokenizer.batch_encode_plus(self.inputs, return_tensors="pt",  max_length=max_length, **encode_plus_kwargs)
                                                   

class KGLMDataset(Dataset):
    def __init__(self, port, db, collection, tokenizer, max_seq_length, neighborhood=False):
        self.client = MongoClient('localhost', port)
        self.db_name = db
        self.collection_name = collection
        self.collection = self.client[db][collection]
        self.tokenizer = tokenizer
        self.length = self.client[self.db_name].command("collstats", self.collection_name)['count']
        self.neighborhood = neighborhood
        self.max_seq_length = max_seq_length

        if self.neighborhood:
            logger.info("Using neighborhoods in inputs...")

    def  __getitem__(self, idx):
        item = {}
        doc = self.collection.find_one({'_id': idx})
        
        if self.neighborhood:
            item["input"] = doc['verbalization']
        else:
            verbalization = doc['verbalization']
            inp = '[SEP]'.join(verbalization.split('[SEP]')[:2])
            item["input"] = inp
            
        item["outputs"] = doc['verbalized_tail']
        item['relation'] = doc['relation']
        item['head'] = doc['head']
        return item
        
    def __len__(self):
        return self.length

    def _collate_eval_with_input_strings(self, items):
        return EvalBatch(items, self.tokenizer, self.max_seq_length)


class Args:
    def __init__(self, length_penalty=1.0, max_output_length=512, length_normalization=0,
                 batch_size=1, beam_size=1, save_file=None, num_predictions=50):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.save_file = save_file
        self.num_predictions = num_predictions
        self.length_penalty = length_penalty
        self.max_output_length = max_output_length
        self.length_normalization = length_normalization


def grouper(arr, n):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    total = len(arr)
    if total % n != 0:
        raise ValueError('Cannot divide %d by %d' % (total, n))
    out = []
    for i in range(int(total/n)):
        start_id = i * n
        out.append(arr[start_id:start_id+n])
    return out

def getScores(ids, scores, pad_token_id, length_normalization):
    # ids is list of tokenized string ids
    # scores is a list of tensors. each tensor contains score of each token in vocab

    # stack scores
    scores = torch.stack(scores, dim=1)
    # after stacking, shape is (batch_size*num_return_sequences, num tokens in sequence, vocab size)

    # normalizing along vocabulary dimension to get probabilities
    log_probs = torch.log_softmax(scores, dim=2)

    # remove start token
    ids = ids[:,1:]

    # gather needed probs

    # x and log_probs should have the same shape
    x = ids.unsqueeze(-1).expand(log_probs.shape)

    # getting probabilities of each token persisting in each sequence 
    needed_logits = torch.gather(log_probs, 2, x)

    # there are probabilities for the same token along 0 axis, so we need to have tensor [seq_num, seq_len]
    final_logits = needed_logits[:, :, 0]

    # removing paddings to not include them in prob calculation
    padded_mask = (ids == pad_token_id)
    final_logits[padded_mask] = 0

    # summing probabilities up along seq_len dimension to get probability of each sequence
    final_scores = final_logits.sum(dim=-1)
    
    if length_normalization == 1:
        sequence_lengths = torch.sum(~padded_mask, dim=1)
        final_scores = final_scores/sequence_lengths

    return final_scores.cpu().detach().numpy()


def eval(model, dataset, args):
    logger.info('Using model.generate')

    data_loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=1,
        collate_fn=dataset._collate_eval_with_input_strings)
    
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    targets = []
    predictions = []
    prediction_scores = []
    model_inputs = []
    relations = []
    heads = []
    
    model.eval()
    with torch.inference_mode():
        for steps, batch in enumerate(loader):
            
            input_ids, attention_mask, target_text, input_text, relation, head = batch.inputs_tokenized.input_ids, \
                batch.inputs_tokenized.attention_mask, batch.target_text, batch.inputs, batch.relations, batch.heads
                
            outputs = model.generate(input_ids = input_ids.cuda(), attention_mask=attention_mask.cuda(),
                        temperature=1.0,
                        do_sample=True,
                        num_return_sequences = args.num_predictions,
                        num_beams=args.beam_size,
                        eos_token_id = dataset.tokenizer.eos_token_id,
                        pad_token_id = dataset.tokenizer.pad_token_id,
                        output_scores = True,
                        return_dict_in_generate=True,
                        length_penalty = args.length_penalty,
                        max_length=args.max_output_length,
                        )
            sequences = outputs.sequences
            
            if args.beam_size > 1:
                final_scores = outputs.sequences_scores
                if args.length_penalty == 1:
                    # get sequence lengths. see getScores for how this works
                    sequence_lengths = torch.sum((sequences[:,1:] != dataset.tokenizer.pad_token_id), dim=1)
                    final_scores = final_scores/sequence_lengths
                final_scores = final_scores.cpu()

            else:
                scores = outputs.scores
                final_scores = getScores(sequences, scores, dataset.tokenizer.pad_token_id,
                                         length_normalization = args.length_normalization)
            
            predicted_text = dataset.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            
            if len(predicted_text) == len(input_text):
                final_scores = final_scores.tolist()
            else:
                predicted_text = grouper(predicted_text, args.num_predictions) 
                final_scores = grouper(final_scores, args.num_predictions)
            
            relations.extend(relation)
            heads.extend(head)
            targets.extend(target_text)
            model_inputs.extend(input_text)
            predictions.extend(predicted_text)
            prediction_scores.extend(final_scores)

    correct = 0     
    for p, t in zip(predictions, targets):
        if t in p:
            correct += 1
            

    data_to_save = {'prediction_strings': predictions, 
                    'scores': prediction_scores,
                    'target_strings': targets,
                    'input_strings': model_inputs,
                    'relations': relations, 
                    'heads': heads}


    scores_dir = 'scores/'
    if not os.path.isdir(scores_dir):
        os.makedirs(scores_dir)

    fname = os.path.join(scores_dir, args.save_file + '.pickle')

    pickle.dump(data_to_save, open(fname, 'wb'))
    accuracy = correct / len(targets)
    return accuracy    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# gathering all known links for specific head and relation
def get_filtering_entities(head, relation, collection_names=[args.train_collection_input, args.valid_collection_input, args.test_collection_input], inverse=False):
    client = MongoClient('localhost', args.mongodb_port)
    all_filter_entities = []

    for coll in collection_names:
        collection = client[args.input_db][coll]
        filter_entities = []

        if inverse:
            for doc in collection.find({'tail': head, 'relation': relation}):
                filter_entities.append(entity_mapping[doc['head']])
        else:
            for doc in collection.find({'head': head, 'relation': relation}):
                filter_entities.append(entity_mapping[doc['tail']])
        
        all_filter_entities.extend(filter_entities)
    
    return all_filter_entities

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

logger.info("Loading mappings...")
entity_mapping = {}
inverse_entity_mapping = {}

with open(args.entity_mapping_path, "r") as f:
    for line in tqdm(f, total=4818679):
        line = line.strip().split("\t")
        id_, name = line[0], line[1]
        entity_mapping[id_] = name
        inverse_entity_mapping[name] = id_


inverse_relation_mapping = {}

with open(args.relation_mapping_path, "r") as f:
    for line in tqdm(f, total=828):
        line = line.strip().split("\t")
        id_, relation = line[0], line[1]
        inverse_relation_mapping[relation] = id_



if args.cpt_path:
    save_file = 'scores_{}'.format(args.cpt_path.replace("/", "_"))
else:
    save_file = 'scores_{}'.format(args.from_hub.replace("/", "_"))

torch.cuda.set_device(args.gpu)

if args.cpt_path and args.model_cfg and args.tokenizer:

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    model = T5ForConditionalGeneration(config=model_cfg)
    model_cpt = os.path.join(args.cpt_path, 'model_best.pth')
    cpt = torch.load(model_cpt, map_location='cpu')
    model.load_state_dict(cpt['model_state_dict'])

elif args.from_hub:
    tokenizer = AutoTokenizer.from_pretrained(
        args.from_hub,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.from_hub,
    )
else:
    logger.error("You must specify arguments for the model")


model.to('cuda:{}'.format(args.gpu))
model.eval()

tokenizer.add_special_tokens({'sep_token': '[SEP]'})

model_args = Args(save_file=save_file)
dataset = KGLMDataset(args.mongodb_port, args.input_db, args.verbalized_eval_collection, tokenizer, max_seq_length=model_args.max_output_length, neighborhood=args.neighborhood)

acc = eval(model, dataset, model_args) 
logger.info("Accuracy@all: ", acc)

scores_data = pickle.load(open(os.join('scores/', model_args.save_file + '.pickle'), 'rb'))

# creating list of dictionaries {prediction: score} for each input 
# and leaving only existing entities from KG in case of transductive setting
predictions_scores_dicts = []

if args.transductive and args.filter_unknown_entities:
    logger.info("Filtering unknown entities...")

for prediction_arr, score_arr in tqdm(zip(scores_data['prediction_strings'], scores_data['scores']), total=len(scores_data['prediction_strings'])):
    ps_pairs = [(p, s) for p, s in zip(prediction_arr, score_arr)]
    ps_pairs = list(set(ps_pairs)) # while sampling, duplicates are created
    ps_dict_only_entities = defaultdict(list)

    if args.transductive and args.filter_unknown_entities:

        # remove predictions that are not entities 
        for ps in ps_pairs:
            if ps[0] in inverse_entity_mapping:
                ps_dict_only_entities[ps[0]] = ps[1]

        predictions_scores_dicts.append(ps_dict_only_entities)

    else:
        for ps in ps_pairs:
            ps_dict_only_entities[ps[0]] = ps[1]

        predictions_scores_dicts.append(ps_dict_only_entities)


# fitering predictions 
predictions_filtered = []

if args.filter_known_links:
    logger.info("Filtering known links...")

for i in tqdm(range(len(predictions_scores_dicts))):
    ps_dict = predictions_scores_dicts[i].copy()
    target = scores_data['target_strings'][i]
    inputs = scores_data['input_strings'][i]
    relation = scores_data['relations'][i]
    head = scores_data['heads'][i]
    prediction_strings = ps_dict.keys()
    
    # getting all tails connected with this input

    if args.filter_known_links:

        if target in prediction_strings:
            original_score = ps_dict[target]

        inverse = 'inverse of' in relation
        if inverse:
            relation = relation.replace("inverse of", "").strip()
            relation = inverse_relation_mapping[relation]
            filtering_entities = get_filtering_entities(head, relation, inverse=True)

        else:
            relation = inverse_relation_mapping[relation]
            filtering_entities = get_filtering_entities(head, relation, inverse=False)

        # if there is a link between predicted and input entities, we don't consider it during ranking   
        for ent in filtering_entities:
            if ent in ps_dict:
                ps_dict[ent] = -float("inf")

        if target in prediction_strings:
            ps_dict[target] = original_score
        

    # dividing scores and predictions to normalize the scores    
    names_arr = []
    scores_arr = []
    for k, v in ps_dict.items():
        names_arr.append(k)
        scores_arr.append(v)
    
    # normalizing scores
    scores_arr = np.array(scores_arr)
    # tbd check the correctness of softmax
    scores_arr = softmax(scores_arr)
    
    # forming new list of dictionaries {prediction: normalized_score}
    for name, score in zip(names_arr, scores_arr):
        ps_dict[name] = score
    
    predictions_filtered.append(ps_dict)


# calculating metrics
count = {}
reciprocal_ranks = 0.0
k_list = [1, 3, 5, 10]

for k in k_list:
    count[k] = 0
    
total_count = 0

for i in tqdm(range(len(predictions_filtered))):
    
    target = scores_data['target_strings'][i]
    ps_dict = predictions_filtered[i]
    
    ps_sorted = sorted(ps_dict.items(), key=lambda item: -item[1])
    inputs = scores_data['input_strings'][i]

    if len(ps_dict) == 0:
        preds = []
    else:
        preds = [x[0] for x in ps_sorted]

    if target in preds:
        rank = preds.index(target) + 1
        reciprocal_ranks += 1./rank
        
    for k in k_list:
        if target in preds[:k]:
            count[k] += 1
        
total_count = len(predictions_filtered)


with open(os.join('scores/', args.output_file),'a') as f:
    f.write('Scored model: {} \n'.format(model_args.save_file))
    logger.info('Scored model: {} \n'.format(model_args.save_file))
    f.write('Acc: {} \n'.format(acc))
    logger.info('Acc: {} \n'.format(acc))

    for k in k_list:
        hits_at_k = count[k] / total_count
        logger.info('hits@{}'.format(k), hits_at_k)
        f.write('hits@{}: {} \n'.format(k, hits_at_k))

    logger.info('mrr', reciprocal_ranks/total_count)
    f.write('MRR: {} \n \n'.format(reciprocal_ranks / total_count))
