import pandas as pd
from tqdm import tqdm
import os
import json
from pykeen import datasets
import time

import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity

import argparse, sys
from loguru import logger


class Verbalizer:
    def __init__(self, base_dataset_collection, similarity_matrix=None, relation2index=None, entity2text=None, relation2text=None):
        
        self.base_dataset_collection = base_dataset_collection

        self.similarity_matrix = similarity_matrix
        self.relation2index = relation2index

        self.entity2text = entity2text
        self.relation2text = relation2text
        
        self.sep = '[SEP]'


    def get_neighbourhood(self, node_id, relation_id=None, tail_id=None, limit=None):

        neighs = []

        # relation_id and tail_id excluded for test dataset
        if tail_id == None:
            cursor = self.base_dataset_collection.find({'head': node_id}, {'_id': False})
            cursor = cursor.limit(limit) if limit else cursor
        
            for doc in cursor:
                neighs.append(doc)

            cursor = self.base_dataset_collection.find({'tail': node_id}, {'_id': False})
            cursor = cursor.limit(limit) if limit else cursor
            for doc in cursor:
                doc['relation'] = 'inverse of ' + doc['relation']
                neighs.append(doc)

        # relation_id and tail_id included for train dataset verbalization to hide the target node
        else:
            cursor = self.base_dataset_collection.find({'head': node_id,
                                                        "$or": [{'tail': {'$ne': tail_id}}, {'relation': {'$ne': relation_id}}]}, {'_id': False})
            cursor = cursor.limit(limit) if limit else cursor
            for doc in cursor:
                neighs.append(doc)

            cursor = self.base_dataset_collection.find({'tail': node_id, 
                                                        "$or": [{'head': {'$ne': tail_id}}, {'relation': {'$ne': relation_id}}]}, {'_id': False})
            cursor = cursor.limit(limit) if limit else cursor
            for doc in cursor:
                doc['relation'] = 'inverse of ' + doc['relation']
                neighs.append(doc)

        return neighs

    
    def verbalize(self, head, relation, tail=None, inverse=False):
        
        relation_prefix = 'inverse of ' if inverse else ''

        limit = 200 if inverse else None

        neighbourhood = self.get_neighbourhood(head, relation, tail, limit)

        relation = relation_prefix + self.relation2text[relation]

        neighbourhood.sort(key=lambda x: 
            (self.similarity_matrix[self.relation2index[self.relation2text[x['relation']]]][self.relation2index[relation]]),
            reverse=True)

        neighbourhood = neighbourhood[:512]
        verbalization = "predict {} {} {} {} ".format(self.sep, self.entity2text[head], relation, self.sep)
        
        verbalization += " ".join(list(map(lambda x: self.relation2text[x['relation']] + " " + (self.entity2text[x['tail']] 
            if x['head'] == head else self.entity2text[x['head']]) + " {}".format(self.sep),
             neighbourhood)))
        

        return " ".join(verbalization.split()).strip() 
    

def verbalize_dataset(input_df, output_collection, verbalizer, insert_index=10000):
    # logger.info('Started verbalizing {}th triplet'.format(i))
    start_time = time.time()
    docs = []

    for i, doc in tqdm(input_df.iterrows(), total=len(input_df)):
        try:
            
            direct_verbalization = verbalizer.verbalize(doc['head'], doc['relation'], doc['tail'])

            docs.append({'_id': i * 2 , 'verbalization': direct_verbalization, 
                                    'head': doc['head'], 'tail': doc['tail'], 
                                    'relation': doc['relation'],
                                    'verbalized_tail': verbalizer.entity2text[doc['tail']]
                                    })
            
            inverse_verbalization = verbalizer.verbalize(doc['tail'], doc['relation'], doc['head'], inverse=True)
            docs.append({'_id': i * 2 + 1, 'verbalization': inverse_verbalization, 
                                    'head': doc['tail'], 'tail': doc['head'], 
                                    'relation': "inverse of " +  doc['relation'],
                                    'verbalized_tail': verbalizer.entity2text[doc['head']]
                                    })
        except Exception as e:
            logger.exception('Exception {} on {}th triplet'.format(e, i))
        
        if i % 50000 == 0 and i > 0:
            output_collection.insert_many(docs)
            docs = []
            if i % 100000 == 0:
                logger.info('verbalized {}th triplet | spanned time: {}s'.format(i, int((time.time() - start_time))))

    output_collection.insert_many(docs)

    return output_collection.count_documents({})


def filter_only_train_entities(df, base_df):
    df_unique_nodes = set(list(df["head"]) + list(df["tail"]))
    common_unique_nodes = df_unique_nodes & set(list(base_df["head"]) + list(base_df["tail"]))

    nodes2drop = df_unique_nodes - common_unique_nodes
    return df[(df['tail'].apply(lambda x: x not in nodes2drop)) & (df['head'].apply(lambda x: x not in nodes2drop))].reset_index(drop=True)


parser = argparse.ArgumentParser()

parser.add_argument("--relation_vectors_path", help="path to the embeddings of verbalized relations", default="data/embeddings/fasttext_vecs-wikidata5m.npy")
parser.add_argument("--rel2ind_path", help="path to the mapping of textual relations to the index of corresponding vectors", default="data/relation2ind-wikidata5m.json")
parser.add_argument("--entity_mapping_path", help="path to the entity2text mapping", default="data/mappings/wd5m_aliases_entities_v3.txt")
parser.add_argument("--relation_mapping_path", help="path to the relation2text mapping", default="data/relation2text-wikidata5m.json")
parser.add_argument("--mongodb_port", help="port of the mongodb collection with the dataset", type=int, default=27018)
parser.add_argument("--input_db", help="name of the mongo database that stores wikidata5m dataset", default='wikidata5m')
parser.add_argument("--train_collection_input", help="name of the collection that stores train KG", default='train-set')
parser.add_argument("--valid_collection_input", help="name of the collection that stores valid KG", default='valid-set')
parser.add_argument("--test_collection_input", help="name of the collection that stores test KG", default='test-set')
parser.add_argument("--train_collection_output", help="name of the collection that stores verbalized train KG", default='verbalized_train')
parser.add_argument("--valid_collection_output", help="name of the collection that stores verbalized valid KG", default='verbalized_valid')
parser.add_argument("--test_collection_output", help="name of the collection that stores verbalized test KG", default='verbalized_test')
parser.add_argument("--filter_transductive", help="whether to filter all entities from test and valid KG that are not in the train KG", action='store_true')

args = parser.parse_args()


vecs = np.load(args.relation_vectors_path)
similarity_matrix = cosine_similarity(vecs)

with open(args.rel2ind_path, 'r') as f:
    rel2ind = json.load(f)


entity_mapping = {}

with open(args.entity_mapping_path, 'r') as f:
    for line in tqdm(f):
        line = line.strip().split('\t') 
        _id, name = line[0], line[1]
        entity_mapping[_id] = name


with open(args.relation_mapping_path, 'r') as f:
    relation_mapping = json.load(f)


dataset = datasets.Wikidata5M()
train_df = pd.read_csv(dataset.training_path, sep='\t', names=['head', 'relation', 'tail'], encoding='utf-8')
valid_df = pd.read_csv(dataset.validation_path, sep="\t", names=["head", "relation", "tail"], encoding="utf-8")
test_df = pd.read_csv(dataset.testing_path, sep="\t", names=["head", "relation", "tail"], encoding="utf-8")


logger.info('Length of the train KG: {}'.format(len(train_df)))

if args.filter_transductive:
    original_valid_len = len(valid_df)
    original_test_len = len(test_df)

    logger.info('Filtering valid and test datasets...')

    valid_df = filter_only_train_entities(valid_df, train_df)
    test_df = filter_only_train_entities(test_df, train_df)

    logger.info('There were filtered {} in the valid KG, the length of updated valid KG: {}'.format(original_valid_len - len(valid_df), len(valid_df)))
    logger.info('There were filtered {} in the test KG, the length of updated test KG: {}'.format(original_test_len - len(test_df), len(test_df)))

else:
    logger.info('Length of the valid KG: {}'.format(len(valid_df)))
    logger.info('Length of the test KG: {}'.format(len(test_df)))


client = MongoClient('localhost', args.mongodb_port)
DB_NAME = args.input_db


collection_train = client[DB_NAME][args.train_collection_input]
collection_valid = client[DB_NAME][args.valid_collection_input]
collection_test = client[DB_NAME][args.test_collection_input]

logger.info('Creating indexes in the collection with the KGs...')
for coll in [collection_train, collection_valid, collection_test]:
    coll.create_index([("head", 1)])
    coll.create_index([("tail", 1)])
    coll.create_index([("relation", 1)])

logger.info('Populating collection with the train KG...')
docs = []
for i, doc in tqdm(train_df.iterrows(), total=len(train_df)):
    docs.append({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})
    if i % 100000 == 0 and i > 0:
        collection_train.insert_many(docs)
        docs = []
collection_train.insert_many(docs)


logger.info('Populating collection with the valid KG...')
for i, doc in tqdm(valid_df.iterrows(), total=len(valid_df)):
    collection_valid.insert_one({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})


logger.info('Populating collection with the test KG...')
for i, doc in tqdm(test_df.iterrows(), total=len(test_df)):
    collection_test.insert_one({'_id': i , 'head': doc['head'], 'tail': doc['tail'], 'relation': doc['relation']})


verbalizer_train = Verbalizer(collection_train, similarity_matrix=similarity_matrix,
                                     relation2index=rel2ind, entity2text=entity_mapping, relation2text=relation_mapping)


logger.info('Verbalizing train KG...')
ouput_train_collection = client[DB_NAME][args.train_collection_output]
train_res = verbalize_dataset(train_df, ouput_train_collection, verbalizer_train)
assert train_res == len(train_df) * 2

logger.info('Verbalizing valid KG...')
ouput_valid_collection = client[DB_NAME][args.valid_collection_output]
valid_res = verbalize_dataset(valid_df, ouput_valid_collection, verbalizer_train)
assert valid_res == len(valid_df) * 2

logger.info('Verbalizing test KG...')
ouput_test_collection = client[DB_NAME][args.test_collection_output]
test_res = verbalize_dataset(test_df, ouput_test_collection, verbalizer_train)
assert test_res == len(test_df) * 2