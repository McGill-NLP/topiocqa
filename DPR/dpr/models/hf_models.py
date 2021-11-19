#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, Union

import torch
from torch import Tensor as T
from torch import nn
from transformers.modeling_bert import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

from dpr.models.biencoder import BiEncoder
from dpr.utils.data_utils import Tensorizer
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg, reader=True)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(cfg, tokenizer=None, reader=False):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    truncation = cfg.encoder.truncation

    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )
        if cfg.special_tokens:
            _add_special_tokens(tokenizer, cfg.special_tokens)
    if not reader:
        return BertTensorizer(tokenizer, sequence_length, truncation=truncation)
    else:
        return BertTensorizerReader(tokenizer, sequence_length, truncation=truncation)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 50
    unused_ids = [
        tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)
    ]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        del tokenizer.vocab["[unused{}]".format(idx)]
        tokenizer.vocab[special_tokens[idx]] = id
        tokenizer.ids_to_tokens[id] = special_tokens[idx]

    tokenizer._additional_special_tokens = list(special_tokens)
    logger.info(
        "Added special tokenizer.additional_special_tokens %s",
        tokenizer.additional_special_tokens,
    )
    logger.info("Tokenizer's all_special_tokens %s", tokenizer.all_special_tokens)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case
        )
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim, **kwargs
            )
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True, truncation: Union[bool, str] = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        self.truncation = truncation

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                # max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=self.truncation,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                # max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=self.truncation,
            )
        if len(token_ids) > self.max_length:
            pass
        if title is not None and len(token_ids) > self.max_length:
            # passage is longer than max_length
            logger.info(f"Truncating sequence: {title}: {text[:50]}.... . " + \
                f"Sequence length is {len(token_ids)}, allowed is {self.max_length}")
        if title is None and len(token_ids) > self.max_length:
            sep_token_idxs = [i for i, t in enumerate(token_ids) if t == self.tokenizer.sep_token_id][:-1]
            query_sub_parts = []
            start_idx = 0
            for sep_token_idx in sep_token_idxs:
                end_idx = sep_token_idx + 1
                query_sub_parts.append(token_ids[start_idx:end_idx])
                start_idx = end_idx
            query_sub_parts.append(token_ids[start_idx:])
            query_sub_part_lengths = [len(sub_part) for sub_part in query_sub_parts]
            if len(query_sub_parts) == 1:
                query_sub_parts.append([])

            query_length = len(query_sub_parts[0]) + len(query_sub_parts[-1])
            if query_length > self.max_length:
                logger.info(f"First and last question together exceed the maximum limit - {text.split('[SEP]')[0]} [SEP] {text.split('[SEP]')[-1]}")
                token_ids = query_sub_parts[0] + query_sub_parts[-1]
            else:
                query_sub_parts_idxs_to_add = []
                for i in range(len(query_sub_parts) - 2, 0, -1):
                    if query_sub_part_lengths[i] + query_length < self.max_length:
                        query_sub_parts_idxs_to_add.append(i)
                        query_length += query_sub_part_lengths[i]
                    else:
                        break

                token_ids = list(query_sub_parts[0])
                for idx in reversed(query_sub_parts_idxs_to_add):
                    token_ids += query_sub_parts[idx]

                token_ids += query_sub_parts[-1]

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]

class BertTensorizerReader(BertTensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True, truncation: Union[bool, str] = True
    ):
        super().__init__(tokenizer, max_length, pad_to_max, truncation)

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            # title is the question, text_pair is passage_title (title [SEP] sub_title)
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                # max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=self.truncation,
            )
        else:
            # this is passage content
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                # max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=self.truncation,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        # will take care of this in _concat_pair
        # if len(token_ids) >= seq_len:
        #     # if title and question , then add SEP, otherwise not
        #     token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
        #     token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )
