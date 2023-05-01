#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import os
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from DPR.dpr.data.biencoder_data import RepTokenSelector
from DPR.dpr.data.qa_validation import calculate_matches, calculate_chunked_matches
from DPR.dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from DPR.dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from DPR.dpr.models import init_biencoder_components
from DPR.dpr.models.biencoder import BiEncoder, _select_span_with_token
from DPR.dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from DPR.dpr.utils.data_utils import Tensorizer
from DPR.dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from DPR.dense_retriever import LocalFaissRetriever

logger = logging.getLogger(__name__)
# setup_logger(logger)
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)


class DenseRetrieverForInference():

    def __init__(self, cfg: DictConfig):

        cfg = setup_cfg_gpu(cfg)
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
        )
        
        encoder_path = cfg.encoder_path
        if encoder_path:
            logger.info("Selecting encoder: %s", encoder_path)
            encoder = getattr(encoder, encoder_path)
        else:
            logger.info("Selecting standard question encoder")
            encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(
            encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")

        encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
        prefix_len = len(encoder_prefix)

        logger.info("Encoder state prefix %s", encoder_prefix)
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(encoder_prefix)
        }
        # TODO: long term HF state compatibility fix
        model_to_load.load_state_dict(question_encoder_state, strict=False)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)

        ctx_src = cfg.ctx_datatsets[0]
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        all_passages = {}
        logger.info("Loading data to all passages") 
        ctx_src.load_data_to(all_passages)
        self.all_passages = all_passages
        logger.info("Data loaded to all passages")

        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        logger.info("Index buffer size %d", index_buffer_sz)
        index.init_index(vector_size)
        self.retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

        ctx_files_pattern = cfg.encoded_ctx_files[0]
        index_path = cfg.index_path
        pattern_files = glob.glob(ctx_files_pattern)
        pattern_id_prefix = "wiki:"

        if index_path and index.index_exists(index_path):
            logger.info("Index path: %s", index_path)
            self.retriever.index.deserialize(index_path)
        else:
            logger.info("Reading all passages data from files: %s", pattern_files)
            self.retriever.index_encoded_data(
                pattern_files, index_buffer_sz, path_id_prefixes=[pattern_id_prefix] * len(pattern_files)
            )
            if index_path:
                self.retriever.index.serialize(index_path)
                logger.info("Saved index to %s", index_path)

    
    def get_top_docs(self, question, n_docs):
        # logger.info("Getting top %d docs for question: %s", n_docs, question)
        # dummy_doc = {
        #     "id": "1234",
        #     "title": "dummy title [SEP] summy sub-title",
        #     "text": "dummy text",
        #     "score": 1.0,
        # }
        # return [dummy_doc] * n_docs
        question_tensor = self.retriever.generate_question_vectors([question], query_token=None)

        top_ids_and_scores = self.retriever.get_top_docs(question_tensor.numpy(), n_docs)
        top_ids_and_scores = top_ids_and_scores[0]

        docs = []
        for doc_id, score in zip(top_ids_and_scores[0], top_ids_and_scores[1]):
            doc = self.all_passages[doc_id]
            docs.append(
                {
                    "id": doc_id,
                    "score": score,
                    "title": doc[1],
                    "text": doc[0],
                }
            )
        return docs
