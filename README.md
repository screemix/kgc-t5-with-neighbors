# kgc-t5-with-neighbors

This repository contains the source code of experiments and demonstration of performance of generative models enhanced with KG neighborhood in its input applied to the KGC (Knowledge Graph Completion) task on the WikiData5M dataset.

---

## Reproducing experiments

Launch mongodb container with docker:

```docker run --name mongodb -d -p 27018:27018 -v ~/data:/data/db mongo:latest```

Download mappings and prepare relations' embeddings:

```python3 data-preparation/prepare_relation_embeddings.py```

Prepare verbalizations for training:

```python3 verbalization.py --mongodb_port 27018```

Lauch training:
```cd scripts/```
```CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./train_wikidata5m_parameter_selection.sh```

---

## Demo notebooks

In the notebook ```demo-notebooks/t5_KGC_neighbors_demo.ipynb``` one can find all the necessary steps to prepare KG input to be fed to the T5 model and a way to download and use the model for their own purposes.

The notebook ```demo-notebooks/gpt4_KGC_demo.ipynb``` contains all the steps for data preparation and propmts for an OpenAI ChatGPT applied to KGC tasks on the Wikidata5M dataset and its comparison to our custom T5 model.

More details about experiments are on our blog post - [link].

Also, one can open the notebooks in Google colab:

```t5_KGC_neighbors_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/t5_KGC_neighbors_demo.ipynb#scrollTo=ixe4066dgVbB) 

```gpt4_KGC_demo.ipynb``` - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/screemix/kgc-t5-with-neighbors/blob/main/gpt4_KGC_demo.ipynb)
