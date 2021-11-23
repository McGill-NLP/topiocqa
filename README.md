# TopiOCQA: Open-domain Conversational Question Answering with Topic Switching

This repository contains code and data for reproducing the results of our paper:

Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries, Siva Reddy. [TopiOCQA: Open-domain Conversational Question Answering with Topic Switching](https://arxiv.org/abs/2110.00768).

To interactively explore and download the dataset, please visit the [project page](https://mcgill-nlp.github.io/topiocqa/).

To cite this work, please use the following citation:
```
@inproceedings{adlakha2020topiocqa,
  title={Topi{OCQA}: Open-domain Conversational Question Answering with Topic Switching},
  author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  booktitle={arXiv preprint arXiv:2110.00768},
  year={2020}
}
```

This repository contains the code for the for the folloewing models described in the paper.
1. [DPR](https://github.com/facebookresearch/DPR) (Dense Passage Retrieval)
2. [FiD](https://github.com/facebookresearch/FiD) (Fusion-in-Decoder)

## Resources & Data
All preprocessed data, model checkpoints, trained passage embeddings, and results are available for download using `python download_data.py`. The script is based on `download_data.py` [script](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) in DPR repository. Here is an example:

```
python download_data.py --resource data.retriever.all_history.train
```

To see all options for `--resource`, run `python download_data.py`. By default, the downloaded data is stored in `downloads` directory. To change the directory, use `--output_dir` argument. 

## Modeling Retriever
Both DPR and FiD models use retriever detailed in DPR paper. The retriever is a bi-encoder network that takes in a query and a passage and outputs an embedding for both. The score is computed as the dot product of the query and passage embeddings. TopiOCQA uses three question representation types as illustrated below:



|  Q<sub>1</sub> : <span style="font-weight:normal">who is lead singer of rage against the machine? </span><br /> A<sub>1</sub> :  <span style="font-weight:normal">Zack de la Rocha </span><br/> <br/> Q<sub>2</sub> : <span style="font-weight:normal">when was it formed? </span><br/> A<sub>2</sub> : <span style="font-weight:normal">1991</span> <br/> <br/> Q<sub>3</sub> : <span style="font-weight:normal">was it nominated for any award? </span>|
| :-------------- |
| **Original** :  was it nominated for any award <br/> **AllHistory** : who is lead singer of rage against the machine [SEP] Zack de la Rocha [SEP] when was it formed [SEP] 1991 [SEP] was it nominated for any award <br/> **Rewrites** : was rage against the machine nominated for any award|


For more details on question representations, please see Section 5.1.2 of the [paper](https://arxiv.org/abs/2110.00768).

DPR retriever requires question-passage pairs in the following format for training:

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]

```

We provide TopiOCQA data pre-processed in this format. For `AllHistory` question representation, the dataset can be downloaded with the following command:

```
python download_data.py --resource data.retriever.all_history
```

The retriever model reported in the paper is trained on 4 x 40GB A100 GPU machines, using the following command:
```
python -m torch.distributed.launch --nproc_per_node=4
        DPR/train_dense_encoder.py \
        train_datasets=[topiocqa_train_all_history] \
        dev_datasets=[topiocqa_dev_all_history] \
        train=biencoder_topiocqa \
        output_dir={your output dir}
```
If the downloaded files are kept in the `downloads` directory, please change the `file` parameters in `DPR/conf/datasets/encoder_train_default.yaml` to absolute paths of the downloaded files.

### Retiever Inference

The trained model checkpoint and the inference results can be directly downloaded using the following command:

```
python download_data.py --resource model_checkpoints.retriever.dpr.all_history
python download_data.py --resource results.retriever.dpr.all_history
```

TODO: Passage embeddings

## Modeling Reader
TODO