# kgc-t5-with-neighbors

This repository contains the source code of experiments and demonstration of models' perfrormance from the EMNLP 2023 paper "Better Together: Enhancing Generative Knowledge Graph Completion". 

Additional details about experiments are published on our blog post - [link](https://medium.com/deeppavlov/improving-knowledge-graph-completion-with-generative-lm-and-neighbors-5bee426223c8).

---

This repository is currently under the finilizing process. 
Current todo's:

- [x] prepare code for reproducing verbalization of the Wikidata5m dataset
- [x] prepare scripts to reproduce training and evaluation on the Wikidata5m
- [ ] add instructions for evaluation
- [ ] prepare code for interpretation
- [ ] prepare code for reproducing verbalization of the ILPC dataset
- [ ] prepare scripts to reproduce training and evaluation on the ILPC 


---
## Table of contents

1. Reproducing experiments
2. Demo notebooks

---

## Reproducing experiments

Launch mongodb container with docker:

```docker run --name mongodb -d -p 27018:27018 -v ~/data:/data/db mongo:latest```

Install all requirements:

```pip3 install -r requirements.txt```

Download mappings and prepare relations' embeddings:

```python3 data-preparation/prepare_relation_embeddings.py```

Prepare verbalizations for training:

```python3 verbalization.py --mongodb_port 27018```

Launch training using neighborhood in the context:
```cd scripts/```

```CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./train_wikidata5m.sh```

Launch training without neighborhood in the context:
```cd scripts/```

```CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./train_wikidata5m_baseline.sh```

---

## Demo notebooks

In the notebook ```demo-notebooks/t5_KGC_neighbors_demo.ipynb``` one can find all the necessary steps to prepare KG input to be fed to the T5 model and a way to download and use the model for their own purposes.

The notebook ```demo-notebooks/gpt4_KGC_demo.ipynb``` contains all the steps for data preparation and propmts for an OpenAI ChatGPT applied to KGC tasks on the Wikidata5M dataset and its comparison to our custom T5 model.

Also, one can open the notebooks in Google colab:

```t5_KGC_neighbors_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/t5_KGC_neighbors_demo.ipynb#scrollTo=ixe4066dgVbB) 

```gpt4_KGC_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/gpt4_KGC_demo.ipynb)
