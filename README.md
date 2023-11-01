# kgc-t5-with-neighbors

This repository contains demonstration of performance of generative models enhanced with KG neighborhood in its input applied to the KGC (Knowledge Graph Completion) task on the WikiData5M dataset.

In the notebook ```t5_KGC_neighbors_demo.ipynb``` one can find all the necessary steps to prepare KG input to be fed to the T5 model and a way to download and use the model for their own purposes.

The notebook ```gpt4_KGC_demo.ipynb``` contains all the steps for data preparation and propmts for an OpenAI ChatGPT applied to KGC tasks on the Wikidata5M dataset and its comparison to our custom T5 model.

More details about experiments are on our blog post - [link].

---

Also, one can open the notebooks in Google colab:

```t5_KGC_neighbors_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/t5_KGC_neighbors_demo.ipynb#scrollTo=ixe4066dgVbB) 

```gpt4_KGC_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/gpt4_KGC_demo.ipynb)

Download mappings and prepare relations' embeddings:

```python3 data-preparation/prepare_relation_embeddings.py```

Run verbalization:

```python3 verbalization.py --relation_vectors_path "data/embeddings/fasttext_vecs-wikidata5m.npy" --rel2ind_path "data/relation2ind-wikidata5m.json" --entity_mapping_path "data/mappings/wd5m_aliases_entities_v3.txt" --relation_mapping_path "data/relation2text-wikidata5m.json" --mongodb_port 27017```

Launch mongodb container with docker:

```docker run --name mongodb -d -p 27018:27018 -v ~/data:/data/db mongo:latest```
