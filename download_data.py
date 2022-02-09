#!/usr/bin/env python3

# This script is adapted from DPR CC-BY-NC 4.0 licensed repo (https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)

RESOURCES_MAP = {
    "data.wikipedia_split.full_wiki": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/wikipedia_split/full_wiki.jsonl",
        "original_ext": ".jsonl",
        "desc": "Entire wikipedia corpus",
    },
    "data.wikipedia_split.full_wiki_segments": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/wikipedia_split/full_wiki_segments.tsv",
        "original_ext": ".tsv",
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)"
    },
    "data.gold_passages_info.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for all train passages",
    },
    "data.gold_passages_info.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for all dev passages",
    },
    "data.gold_passages_info.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for original train passages",
    },
    "data.gold_passages_info.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for original dev passages",
    },
    "data.gold_passages_info.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for rewrites train passages",
    },
    "data.gold_passages_info.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/gold_passages_info/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for rewrites dev passages",
    },
    "data.retriever.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_all_history_retriever_train.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for all train passages",
    },
    "data.retriever.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_all_history_retriever_dev.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for all dev passages",
    },
    "data.retriever.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_original_retriever_train.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for original train passages",
    },
    "data.retriever.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_original_retriever_dev.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for original dev passages",
    },
    "data.retriever.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_rewrites_t5_qrecc_retriever_train.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for rewrites train passages",
    },
    "data.retriever.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/topiocqa_rewrites_t5_qrecc_retriever_dev.json",
        "original_ext": ".json",
        "desc": "Retriever passages info for rewrites dev passages",
    },
    "data.retriever.qas.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_all_history_retriever_train.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for all train passages",
    },
    "data.retriever.qas.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_all_history_retriever_dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for all dev passages",
    },
    "data.retriever.qas.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_original_retriever_train.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for original train passages",
    },
    "data.retriever.qas.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_original_retriever_dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for original dev passages",
    },
    "data.retriever.qas.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_rewrites_t5_qrecc_retriever_train.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for rewrites train passages",
    },
    "data.retriever.qas.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/data/retriever/qas/topiocqa_rewrites_t5_qrecc_retriever_dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever passages info for rewrites dev passages",
    },
    "passage_embeddings.all_history.wikipedia_passages": {
        "s3_url": [
            "https://topiocqa.s3.us-east-2.amazonaws.com/passage_embeddings/all_history/wikipedia_passages_{}.pkl".format(i)
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passages for all history passages",
    },
    "passage_embeddings.original.wikipedia_passages": {
        "s3_url": [
            "https://topiocqa.s3.us-east-2.amazonaws.com/passage_embeddings/original/wikipedia_passages_{}.pkl".format(i)
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passages for original passages",
    },
    "passage_embeddings.rewrites_t5_qrecc.wikipedia_passages": {
        "s3_url": [
            "https://topiocqa.s3.us-east-2.amazonaws.com/passage_embeddings/rewrites_t5_qrecc/wikipedia_passages_{}.pkl".format(i)
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passages for rewrites passages",
    },
    "model_checkpoints.retriever.dpr.all_history.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/retriever/dpr_retriever_all_history",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoints for all history passages",
    },
    "model_checkpoints.retriever.dpr.original.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/retriever/dpr_retriever_original",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoints for original passages",
    },
    "model_checkpoints.retriever.dpr.rewrites_t5_qrecc.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/retriever/dpr_retriever_rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoints for rewrites passages",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.all_history.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/dpr_retriever/dpr_reader_all_history",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoints for all history passages",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.original.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/dpr_retriever/dpr_reader_original",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoints for original passages",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.rewrites_t5_qrecc.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/dpr_retriever/dpr_reader_rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoints for rewrites passages",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.all_history.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/bm25_retriever/dpr_reader_all_history",
        "original_ext": ".pt",
        "desc": "Reader BM25 model checkpoints for all history passages",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.original.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/bm25_retriever/dpr_reader_original",
        "original_ext": ".pt",
        "desc": "Reader BM25 model checkpoints for original passages",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.rewrites_t5_qrecc.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/bm25_retriever/dpr_reader_rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "DPR Reader model checkpoint for rewrites variant with BM25 Retriever",
    },
    "model_checkpoints.reader.fid.dpr_retriever.all_history.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/dpr_retriever/all_history",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for all history variant with DPR Retriever",
    },
    "model_checkpoints.reader.fid.dpr_retriever.original.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/dpr_retriever/original",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for original variant with DPR Retriever",
    },
    "model_checkpoints.reader.fid.dpr_retriever.rewrites_t5_qrecc.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/dpr_retriever/rewrites_t5_qrecc",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for rewrites variant with DPR Retriever",
    },
    "model_checkpoints.reader.fid.bm25_retriever.all_history.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/bm25_retriever/all_history",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for all history variant with BM25 Retriever",
    },
    "model_checkpoints.reader.fid.bm25_retriever.original.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/bm25_retriever/original",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for original variant with BM25 Retriever",
    },
    "model_checkpoints.reader.fid.bm25_retriever.rewrites_t5_qrecc.checkpoint": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/model_checkpoints/reader/fid/bm25_retriever/rewrites_t5_qrecc",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for rewrites variant with BM25 Retriever",
    },
    "results.retriever.dpr.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for all history passages",
    },
    "results.retriever.dpr.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for all history passages",
    },
    "results.retriever.dpr.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for original passages",
    },
    "results.retriever.dpr.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for original passages",
    },
    "results.retriever.dpr.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for rewrites passages",
    },
    "results.retriever.dpr.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/DPR/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for rewrites passages",
    },
    "results.retriever.bm25.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for all history passages",
    },
    "results.retriever.bm25.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for all history passages",
    },
    "results.retriever.bm25.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for original passages",
    },
    "results.retriever.bm25.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for original passages",
    },
    "results.retriever.bm25.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for rewrites passages",
    },
    "results.retriever.bm25.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/retriever/BM25/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for rewrites passages",
    },
    "results.reader.dpr_reader.dpr_retriever.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for all history passages",
    },
    "results.reader.dpr_reader.dpr_retriever.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for all history passages",
    },
    "results.reader.dpr_reader.dpr_retriever.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for original passages",
    },
    "results.reader.dpr_reader.dpr_retriever.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for original passages",
    },
    "results.reader.dpr_reader.dpr_retriever.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for rewrites passages",
    },
    "results.reader.dpr_reader.dpr_retriever.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/DPR/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Reader DPR Retriever results for rewrites passages",
    },
    "results.reader.dpr_reader.bm25_retriever.all_history.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for all history passages",
    },
    "results.reader.dpr_reader.bm25_retriever.all_history.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for all history passages",
    },
    "results.reader.dpr_reader.bm25_retriever.original.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for original passages",
    },
    "results.reader.dpr_reader.bm25_retriever.original.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for original passages",
    },
    "results.reader.dpr_reader.bm25_retriever.rewrites_t5_qrecc.train": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for rewrites passages",
    },
    "results.reader.dpr_reader.bm25_retriever.rewrites_t5_qrecc.dev": {
        "s3_url": "https://topiocqa.s3.us-east-2.amazonaws.com/results/reader/BM25/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Reader BM25 Retriever results for rewrites passages",
    },
}


def download_resource(
    s3_url: str, original_ext: str, resource_key: str, out_dir: str
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", s3_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1]) 

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] +  original_ext))

    wget.download(s3_url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    return save_root, local_file


def download_file(s3_url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info["s3_url"]

    save_root_dir = None
    data_files = []
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir, local_file = download_resource(
                url,
                download_info["original_ext"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            s3_url,
            download_info["original_ext"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    return data_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print(f"Resource key={k}  :  {v['desc']}")


if __name__ == "__main__":
    main()
