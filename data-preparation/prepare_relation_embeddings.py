import pandas as pd
import os
from tqdm import tqdm
import fasttext
import numpy as np
import json
import logging
import wget
from sh import gunzip


data_dir = 'data'

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

mappings_path = os.path.join(data_dir, 'mappings')
embeddings_path = os.path.join(data_dir, 'embeddings')

if not os.path.isdir(mappings_path):
    os.makedirs(mappings_path)

if not os.path.isdir(embeddings_path):
    os.makedirs(embeddings_path)

wget.download("https://storage.googleapis.com/t5-kgc-colab/data/wd5m_aliases_entities_v3.txt", out=mappings_path)
wget.download("https://storage.googleapis.com/t5-kgc-colab/data/wd5m_aliases_relations_v3.txt", out=mappings_path)
wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz", out=embeddings_path)

embeddings_path = os.path.join(embeddings_path, 'cc.en.300.bin.gz')
gunzip(embeddings_path)

relation_mapping_path = os.path.join(mappings_path, 'wd5m_aliases_relations_v3.txt')
relation_mapping = {}

with open(relation_mapping_path, 'r') as f:
    for line in tqdm(f):
        line = line.strip().split('\t') 
        id_, name = line[0], line[1]
        relation_mapping[id_] = name
        relation_mapping["inverse of " + id_] = name

relations = list(set(relation_mapping.values()))

relation2index = {}
for i, rel in enumerate(relations):
    relation2index[rel] = i
    relation2index["inverse of " + rel] = i


with open('data/relation2ind-wikidata5m.json', 'w') as f:
    json.dump(relation2index, f)

for rel in relation_mapping:
    if "inverse of " in rel:
        relation_mapping[rel] = "inverse of " + relation_mapping[rel]
        
with open('data/relation2text-wikidata5m.json', 'w') as f:
    json.dump(relation_mapping, f)

model_en = fasttext.load_model('data/embeddings/cc.en.300.bin')

fasttext_emb = list(map(lambda x: model_en.get_sentence_vector(x), relations))

np.save("data/embeddings/fasttext_vecs-wikidata5m.npy", fasttext_emb)