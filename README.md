# TopiOCQA: Open-domain Conversational Question Answering with Topic Switching

[![arxiv](https://img.shields.io/badge/arXiv-2110.00768-b31b1b.svg)](https://arxiv.org/abs/2110.00768)

This repository contains code and data for reproducing the results of our paper:

Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries, Siva Reddy. [TopiOCQA: Open-domain Conversational Question Answering with Topic Switching](https://arxiv.org/abs/2110.00768).

To download and interactively explore the dataset, please visit the [project page](https://mcgill-nlp.github.io/topiocqa/).

To cite this work, please use the following citation:
```
@article{adlakha2022topiocqa,
  title={Topi{OCQA}: Open-domain Conversational Question Answering with Topic Switching},
  author={Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  journal={Transactions of the Association for Computational Linguistics},
  volume = {10},
  pages = {468-483},
  year = {2022},
  month = {04},
  year={2022},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00471},
  url = {https://doi.org/10.1162/tacl\_a\_00471},
  eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00471/2008126/tacl\_a\_00471.pdf},
}
```

This repository contains the code for the following models described in the paper.
1. [DPR](https://github.com/facebookresearch/DPR) (Dense Passage Retrieval)
2. [FiD](https://github.com/facebookresearch/FiD) (Fusion-in-Decoder)

## Resources & Data
All preprocessed data, model checkpoints, trained passage embeddings, and results are available for download using `python download_data.py`. The script is based on `download_data.py` [script](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) in DPR repository. Here is an example:

```
python download_data.py --resource data.retriever.all_history.train
```

To see all options for `--resource`, run `python download_data.py`.

By default, the downloaded data is stored in `downloads` directory. To change the directory, use `--output_dir` argument. 

## Modeling Retriever
Both DPR and FiD models use retriever detailed in DPR paper. The retriever is a bi-encoder network that takes in a query and a passage and outputs an embedding for both. The score is computed as the dot product of the query and passage embeddings. TopiOCQA uses three question representation types as illustrated below:



|  Q<sub>1</sub> : <span style="font-weight:normal">who is lead singer of rage against the machine? </span><br /> A<sub>1</sub> :  <span style="font-weight:normal">Zack de la Rocha </span><br/> <br/> Q<sub>2</sub> : <span style="font-weight:normal">when was it formed? </span><br/> A<sub>2</sub> : <span style="font-weight:normal">1991</span> <br/> <br/> Q<sub>3</sub> : <span style="font-weight:normal">was it nominated for any award? </span>|
| :-------------- |
| **Original** :  was it nominated for any award <br/> **AllHistory** : who is lead singer of rage against the machine [SEP] Zack de la Rocha [SEP] when was it formed [SEP] 1991 [SEP] was it nominated for any award <br/> **Rewrites** : was rage against the machine nominated for any award|


For more details on question representations, please see Section 5.1.2 of the [paper](https://arxiv.org/abs/2110.00768).

The Wikipedia corpus used in our work can be downloaded using the following command:
```
python download_data.py --resource data.wikipedia_split.full_wiki
```
For modeling, each Wikipedia article is chunked into passages. The corpus in the chucked format can be downloaded using `data.wikipedia_split.full_wiki_segments` as the resource key.


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

The retriever model reported in the paper was trained on 4 x 40GB A100 GPU machines, using the following command:
```
python -m torch.distributed.launch --nproc_per_node=4 \
        DPR/train_dense_encoder.py \
        train_datasets=[topiocqa_train_all_history] \
        dev_datasets=[topiocqa_dev_all_history] \
        train=biencoder_topiocqa \
        output_dir={your output dir}
```
If the downloaded files are not kept in the `downloads` directory, please change the `file` parameters in `DPR/conf/datasets/encoder_train_default.yaml` to absolute paths of the dataset files.

### Retiever Inference

The trained model checkpoint for `AllHistory` can also be downloaded with the following command:
```
python download_data.py --resource model_checkpoints.retriever.dpr.all_history
```

Before performing inference, the passage encoder needs to generate embeddings for all passages in the corpus. This is highly parallelizable as each shard of the corpus can be proccessed asynchronously, as explained in [DPR repository](https://github.com/facebookresearch/DPR#retriever-inference). The passage embeddings can be generated using the following command:

```
python DPR/generate_dense_embeddings.py \
        model_file={path to model file} \
        ctx_src=dpr_wiki_topiocqa \
        shard_id={shard} \
        num_shards=50 \
        out_file={output directory + name prefix, e.g. /home/topiocqa/downloads/passage_embeddings/all_history/wikipedia_passages} \
        batch_size=128
```

`{shard}` takes all numeric values between `0` and `49`. Each shard was processed on 2 x 16GB V100 GPU machines. This can also be run on a single machine by reducing the batch size. We provide the generated passage embeddings which can be downloaded using `passage_embeddings.all_history.wikipedia_passages` as the resource key.

For retiever inference, the original DPR codebase uses datasets in CSV format. We provide TopiOCQA data pre-processed in CSV format, which can be downloaded using `data.retriever.qas.all_history` as resource key.

We can now perform retrieval inference. The following command is for inference over the dev set of TopiOCQA:
```
python DPR/dense_retriever.py \
        model_file={path to model file} \
        qa_dataset=topiocqa_dev_all_history \
        ctx_datatsets=[dpr_wiki_topiocqa] \
        encoded_ctx_files=[{list of encoded document files glob expression, e.g. \"/home/topiocqa/downloads/passage_embeddings/all_history/wikipedia_passages_*\"}] \
        out_file={your output file}
``` 

The output file with the retrieved results has the following format:
```
[
    {
        "question": "...",
        "answers": ["..."],
        "ctxs": [
            {
                "id": "...",
                "title": "...",
                "text": "....",
                "score": "...",
                "has_answer": true|false
     },
]
```
Retrieval inference for DPR is computationally expensive as it builds an in-memory index of the entire corpus. For our corpus (~25.7 million passages), the peak RAM consumption was 148GB. The GPU infrastucture used for inference was 4 x 16GB V100 GPU machines. The results from retrieval inference can also be downloaded by using `results.retriever.dpr.all_history` as the resource key.

DPR inference procedure evaluates by checking the presence of answer span in the retrieved passage. This is sub-optimal for TopiOCQA as it is an abstractive question-answering dataset, therefore the answer span may not be present in any passage. TopiOCQA provides the gold question-passage pairs which can be used for evaluation. Given the retriever inference results, the evaluation metrics can be computed by using the following command:
```
python evaluate_retriever.py \
        --data_file {path to data file in JSON format} \
        --results_file {path to retriever results file}
```

## Modeling Reader
We experiment with two reader models - (1) DPR Reader and (2) [FiD](https://github.com/facebookresearch/FiD) (Fusion-in-Decoder). Both reader models directly take the retriever results as input. Additionally, during training, we also provide the gold question-passage pairs.

### DPR Reader

To train the DPR Reader model, we use the following command:
```
python DPR/train_extractive_reader.py \
        encoder.sequence_length=384 \
        train_files={path to the retriever train set results file} \
        dev_files={path to the retriever dev set results file} \
        gold_passages_src={path to gold passage info file for train set} \
        gold_passages_src_dev={path to gold passage info file for train set} \
        output_dir={your output dir}
```

The revelant files to run the above command can be downloaded using the following resource keys: `results.retriever.dpr.all_history`, `data.gold_passages_info.all_history`. First time run will preprocess `train_files` & `dev_files` and convert them into serialized set of .pkl files in the same location and will use them on all subsequent runs. The DPR reader model reported in the paper was trained on 8 x 32GB V100 GPU machines. The trained checkpoint for DPR Reader trained on DPR Retriever results can be downloaded using `model_checkpoints.reader.dpr_reader.dpr_retriever.all_history`.

To evaluate the DPR Reader model, we use the same command as above, but without `train_files` argument:
```
python DPR/train_extractive_reader.py \
        encoder.sequence_length=384 \
        prediction_results_file={your output file path} \
        dev_files={path to the retriever results file} \
        eval_top_docs=[10,20,40,50,80,100] \
        model_file={path to model file} \
        train.dev_batch_size=80 \
        train.log_batch_step=1 \
        passages_per_question_predict=100
```
The inference results can be downloaded using `results.reader.dpr_reader.dpr_retriever.all_history` as the resource key.


### FiD
FiD (Fusion-in-Decoder) is trained using the following command on 8 x 32GB V100 GPU machines.:
```
python -m torch.distributed.launch --nproc_per_node=8 \
        FiD/train_reader.py \
        --model_size base \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 384 \
        --answer_maxlength 100 \
        --per_gpu_batch_size 2 \
        --accumulation_steps 4 \
        --n_context 50 \
        --total_step 15000 \
        --warmup_step 1000 \
        --eval_freq 1000 \
        --checkpoint_dir {your output dir} \
        --train_data {path to the retriever train set results file} \
        --gold_passages_train {path to gold passage info file for train set} \
        --eval_data {path to the retriever dev set results file}
```
The trained checkpoint for FiD trained on DPR Retriever results can be downloaded using `model_checkpoints.reader.fid.dpr_retriever.all_history`. The downloaded file will be a compressed folder which will be required to be extracted before moving to evaluation step.

To evaluate the FiD model, we use the following command on a single 32GB V100 GPU machine:
```
python FiD/test_reader.py \
        --model_path {path to model file} \
        --eval_data {path to the retriever results file} \
        --text_maxlength 384 \
        --answer_maxlength 100 \
        --per_gpu_batch_size 16 \
        --n_context 50 \
        --checkpoint_dir {your output file path} \
        --name {dataset split, e.g. test} \
        --write_results \
        --eval_print_freq 10
```
The inference results can be downloaded using `results.reader.fid.dpr_retriever.all_history` as the resource key.


### Reader Result Evaluation
Evaluation in TopiOCQA is different from that performed in original implementation of DPR (Refer to Section 5.2 in the [paper](https://arxiv.org/pdf/2110.00768.pdf)). Our evaluation code is based on [CoQA](https://stanfordnlp.github.io/coqa/). We use the following command to evaluate the reader results:
```
python evaluate_reader.py \
        --data_file {path to train/dev split of the dataset} \
        --results_file {path to reader results file}
```

---

## Contact
For queries and clarifications please contact **vaibhav.adlakha (at) mila (dot) quebec**

---

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
