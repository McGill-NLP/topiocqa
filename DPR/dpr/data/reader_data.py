#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for the Reader model related data processing tasks
"""

import collections
import glob
import json
import logging
import math
import multiprocessing
import os
import pickle
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from functools import partial
from typing import Tuple, List, Dict, Iterable, Optional
from torch import Tensor as T
from tqdm import tqdm

from dpr.utils.data_utils import Tensorizer, read_serialized_data_from_files
from dpr.utils.span_heuristic import find_closest_span_match

logger = logging.getLogger()

F1_THRESHOLD = 0.5


class ReaderPassage(object):
    """
    Container to collect and cache all Q&A passages related attributes before generating the reader input
    """

    def __init__(
        self,
        id=None,
        text: str = None,
        title: str = None,
        score=None,
        has_answer: bool = None,
        answers: List[str] = None,
        extra_text: str = "",
    ):
        self.id = id
        # string passage representations
        assert extra_text != ""
        self.passage_text = extra_text + " " + text
        self.title = title
        assert "[SEP]" in self.title
        self.score = score
        self.has_answer = has_answer
        self.answers_spans = None
        self.closest_span = None
        if answers:
            self.has_answer, self.closest_span = self.get_span_from_answers(answers, self.passage_text)
        self.passage_token_ids = None
        # offset of the actual passage (i.e. not a question or may be title) in the sequence_ids
        self.passage_offset = None
        # passage token ids
        self.sequence_ids = None

    def on_serialize(self):
        # store only final sequence_ids and the ctx offset
        self.sequence_ids = self.sequence_ids.numpy()
        # TODO: revert this back after understanding how multiprocessing works
        # self.passage_text = None
        # self.title = None
        # self.passage_token_ids = None

    def on_deserialize(self):
        self.sequence_ids = torch.tensor(self.sequence_ids)
    
    def get_span_from_answers(self, answers: List[str], passage_text: str) -> Tuple[bool, List[Tuple[int, int]]]:
        assert len(answers) == 1
        answer = answers[0]
        closest_span, score = find_closest_span_match(passage_text, answer)
        if score > F1_THRESHOLD:
            return True, closest_span
        return False, None


class ReaderSample(object):
    """
    Container to collect all Q&A passages data per singe question
    """

    def __init__(
        self,
        question: str,
        conv_id: str,
        turn_id: str,
        answers: List,
        positive_passages: List[ReaderPassage] = [],
        negative_passages: List[ReaderPassage] = [],
        passages: List[ReaderPassage] = [],
    ):
        self.question = question
        self.answers = answers
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages
        self.passages = passages
        self.conv_id = conv_id
        self.turn_id = turn_id

    def on_serialize(self):
        for passage in self.passages + self.positive_passages + self.negative_passages:
            passage.on_serialize()

    def on_deserialize(self):
        for passage in self.passages + self.positive_passages + self.negative_passages:
            passage.on_deserialize()


class ExtractiveReaderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files: str,
        is_train: bool,
        gold_passages_src: str,
        tensorizer: Tensorizer,
        run_preprocessing: bool,
        num_workers: int,
        encoder_seq_length: int,
    ):
        self.files = files
        self.data = []
        self.is_train = is_train
        self.gold_passages_src = gold_passages_src
        self.tensorizer = tensorizer
        self.run_preprocessing = run_preprocessing
        self.num_workers = num_workers
        self.encoder_seq_length = encoder_seq_length

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_data(
        self,
    ):
        data_files = glob.glob(self.files)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = self._get_preprocessed_files(data_files)
        self.data = read_serialized_data_from_files(preprocessed_data_files)

    def _get_preprocessed_files(
        self,
        data_files: List,
    ):

        serialized_files = [file for file in data_files if file.endswith(".pkl")]
        if serialized_files:
            return serialized_files
        assert len(data_files) == 1, "Only 1 source file pre-processing is supported."

        # data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace(".json", "")
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if serialized_files:
            logger.info("Found preprocessed files. %s", serialized_files)
            return serialized_files

        logger.info(
            "Data are not preprocessed for reader training. Start pre-processing ..."
        )

        # start pre-processing and save results
        def _run_preprocessing(tensorizer: Tensorizer, encoder_seq_length: int):
            # temporarily disable auto-padding to save disk space usage of serialized files
            tensorizer.set_pad_to_max(False)
            serialized_files = convert_retriever_results(
                self.is_train,
                data_files[0],
                out_file_prefix,
                self.gold_passages_src,
                self.tensorizer,
                num_workers=self.num_workers,
                encoder_seq_length=encoder_seq_length,
            )
            tensorizer.set_pad_to_max(True)
            return serialized_files

        if self.run_preprocessing:
            serialized_files = _run_preprocessing(self.tensorizer, self.encoder_seq_length)
            # TODO: check if pytorch process group is initialized
            # torch.distributed.barrier()
        else:
            # torch.distributed.barrier()
            serialized_files = _find_cached_files(data_files[0])
        return serialized_files


SpanPrediction = collections.namedtuple(
    "SpanPrediction",
    [
        "prediction_text",
        "span_score",
        "relevance_score",
        "passage_index",
        "passage_token_ids",
    ],
)

# configuration for reader model passage selection
ReaderPreprocessingCfg = collections.namedtuple(
    "ReaderPreprocessingCfg",
    [
        "use_tailing_sep",
        "skip_no_positves",
        "include_gold_passage",
        "gold_page_only_positives",
        "max_positives",
        "max_negatives",
        "min_negatives",
        "max_retriever_passages",
        "extra_text",
    ],
)

DEFAULT_PREPROCESSING_CFG_TRAIN = ReaderPreprocessingCfg(
    use_tailing_sep=False,
    skip_no_positves=True,
    include_gold_passage=False,
    gold_page_only_positives=True,
    max_positives=20,
    max_negatives=50,
    min_negatives=150,
    max_retriever_passages=200,
    extra_text="",
)

TOPIOCQA_PREPROCESSING_CFG_TRAIN = ReaderPreprocessingCfg(
    use_tailing_sep=False,
    skip_no_positves=True,
    include_gold_passage=True,
    gold_page_only_positives=True,
    max_positives=20,
    max_negatives=50,
    min_negatives=150,
    max_retriever_passages=200,
    extra_text="UNANSWERABLE Yes No",
)

TOPIOCQA_PREPROCESSING_CFG_TEST = ReaderPreprocessingCfg(
    use_tailing_sep=False,
    skip_no_positves=True,
    include_gold_passage=False,
    gold_page_only_positives=True,
    max_positives=20,
    max_negatives=50,
    min_negatives=150,
    max_retriever_passages=200,
    extra_text="UNANSWERABLE Yes No",
)

DEFAULT_EVAL_PASSAGES = 100


def preprocess_retriever_data(
    samples: List[Dict],
    gold_info_file: Optional[str],
    tensorizer: Tensorizer,
    cfg: ReaderPreprocessingCfg = TOPIOCQA_PREPROCESSING_CFG_TRAIN,
    is_train_set: bool = True,
    encoder_seq_length: int = 256,
) -> Iterable[ReaderSample]:
    """
    Converts retriever results into reader training data.
    :param samples: samples from the retriever's json file results
    :param gold_info_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param cfg: ReaderPreprocessingCfg object with positive and negative passage selection parameters
    :param is_train_set: if the data should be processed as a train set
    :return: iterable of ReaderSample objects which can be consumed by the reader model
    """
    sep_tensor = tensorizer.get_pair_separator_ids()  # separator can be a multi token

    if gold_info_file is None:
        cfg = TOPIOCQA_PREPROCESSING_CFG_TEST
    gold_passage_map, canonical_questions = (
        _get_gold_ctx_dict(gold_info_file, cfg.extra_text) if gold_info_file else ({}, {})
    )

    no_positive_passages = 0
    positives_from_gold = 0

    def create_reader_sample_ids(sample: ReaderPassage, question: str, encoder_seq_length: int):
        question_and_title = tensorizer.text_to_tensor(
            sample.title, title=question, add_special_tokens=True
        )
        if sample.passage_token_ids is None:
            sample.passage_token_ids = tensorizer.text_to_tensor(
                sample.passage_text, add_special_tokens=False
            )

        all_concatenated, shift = _concat_pair(
            question_and_title,
            sample.passage_token_ids,
            tailing_sep=sep_tensor if cfg.use_tailing_sep else None,
            encoder_seq_length=encoder_seq_length,
        )
        # all_concatenated should not be more than encoder length
        sample.sequence_ids = all_concatenated
        sample.passage_offset = shift
        assert shift > 1
        if sample.has_answer and is_train_set:
            sample.answers_spans = [
                (span[0] + shift, span[1] + shift) for span in sample.answers_spans
            ]
        return sample

    for sample in samples:
        question = sample["question"].replace("'", "’")
        # for some reason, only this training example was facing an issue
        if question == 'product recall is an episode from which series':
            question = '"product recall" is an episode from which series'
        if is_train_set and question not in canonical_questions:
            logger.info("Question not found in canonical questions: %s", question)
            assert False
        # if question in canonical_questions:
        if is_train_set:
            question = canonical_questions[question]
            assert question == canonical_questions[question]

        positive_passages, negative_passages = _select_reader_passages(
            sample,
            question,
            tensorizer,
            gold_passage_map,
            cfg.gold_page_only_positives,
            cfg.max_positives,
            cfg.max_negatives,
            cfg.min_negatives,
            cfg.max_retriever_passages,
            cfg.include_gold_passage,
            cfg.extra_text,
            is_train_set,
        )
        # create concatenated sequence ids for each passage and adjust answer spans
        positive_passages = [
            create_reader_sample_ids(s, question, encoder_seq_length) for s in positive_passages
        ]
        negative_passages = [
            create_reader_sample_ids(s, question, encoder_seq_length) for s in negative_passages
        ]

        if is_train_set and len(positive_passages) == 0:
            no_positive_passages += 1
            if cfg.skip_no_positves:
                continue

        if next(iter(ctx for ctx in positive_passages if ctx.score == -1), None):
            positives_from_gold += 1

        if is_train_set:
            yield ReaderSample(
                question,
                str(sample["conv_id"]),
                str(sample["turn_id"]),
                sample["answers"],
                positive_passages=positive_passages,
                negative_passages=negative_passages,
            )
        else:
            yield ReaderSample(question, str(sample["conv_id"]), str(sample["turn_id"]), sample["answers"], passages=negative_passages)

    logger.info("no positive passages samples: %d", no_positive_passages)
    logger.info("positive passages from gold samples: %d", positives_from_gold)


def convert_retriever_results(
    is_train_set: bool,
    input_file: str,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    num_workers: int = 8,
    encoder_seq_length: int = 256,
) -> List[str]:
    """
    Converts the file with dense retriever(or any compatible file format) results into the reader input data and
    serializes them into a set of files.
    Conversion splits the input data into multiple chunks and processes them in parallel. Each chunk results are stored
    in a separate file with name out_file_prefix.{number}.pkl
    :param is_train_set: if the data should be processed for a train set (i.e. with answer span detection)
    :param input_file: path to a json file with data to convert
    :param out_file_prefix: output path prefix.
    :param gold_passages_file: optional path for the 'gold passages & questions' file. Required to get best results for NQ
    :param tensorizer: Tensorizer object for text to model input tensors conversions
    :param num_workers: the number of parallel processes for conversion
    :return: names of files with serialized results
    """
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.loads("".join(f.readlines()))
    logger.info(
        "Loaded %d questions + retrieval results from %s", len(samples), input_file
    )
    workers = multiprocessing.Pool(num_workers)
    ds_size = len(samples)
    step = max(math.ceil(ds_size / num_workers), 1)
    chunks = [samples[i : i + step] for i in range(0, ds_size, step)]
    chunks = [(i, chunks[i]) for i in range(len(chunks))]

    logger.info("Split data into %d chunks", len(chunks))

    processed = 0
    _parse_batch = partial(
        _preprocess_reader_samples_chunk,
        out_file_prefix=out_file_prefix,
        gold_passages_file=gold_passages_file,
        tensorizer=tensorizer,
        is_train_set=is_train_set,
        encoder_seq_length=encoder_seq_length,
    )
    serialized_files = []
    for file_name in workers.map(_parse_batch, chunks):
        processed += 1
        serialized_files.append(file_name)
        logger.info("Chunks processed %d", processed)
        logger.info("Data saved to %s", file_name)
    logger.info("Preprocessed data stored in %s", serialized_files)
    return serialized_files


def get_best_spans(
    tensorizer: Tensorizer,
    start_logits: List,
    end_logits: List,
    ctx_ids: List,
    max_answer_length: int,
    passage_idx: int,
    relevance_score: float,
    top_spans: int = 1,
) -> List[SpanPrediction]:
    """
    Finds the best answer span for the extractive Q&A model
    """
    scores = []
    for (i, s) in enumerate(start_logits):
        for (j, e) in enumerate(end_logits[i : i + max_answer_length]):
            scores.append(((i, i + j), s + e))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    chosen_span_intervals = []
    best_spans = []

    for (start_index, end_index), score in scores:
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= max_answer_length

        if any(
            [
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ]
        ):
            continue

        # extend bpe subtokens to full tokens
        start_index, end_index = _extend_span_to_full_words(
            tensorizer, ctx_ids, (start_index, end_index)
        )

        predicted_answer = tensorizer.to_string(ctx_ids[start_index : end_index + 1])
        best_spans.append(
            SpanPrediction(
                predicted_answer, score, relevance_score, passage_idx, ctx_ids
            )
        )
        chosen_span_intervals.append((start_index, end_index))

        if len(chosen_span_intervals) == top_spans:
            break
    return best_spans


def _select_reader_passages(
    sample: Dict,
    question: str,
    tensorizer: Tensorizer,
    gold_passage_map: Dict[str, ReaderPassage],
    gold_page_only_positives: bool,
    max_positives: int,
    max1_negatives: int,
    max2_negatives: int,
    max_retriever_passages: int,
    include_gold_passage: bool,
    extra_text: str,
    is_train_set: bool,
) -> Tuple[List[ReaderPassage], List[ReaderPassage]]:
    answers = sample["answers"]

    ctxs = [ReaderPassage(**ctx, answers=answers, extra_text=extra_text) for ctx in sample["ctxs"]][0:max_retriever_passages]
    answers_token_ids = [
        tensorizer.text_to_tensor(a, add_special_tokens=False) for a in answers
    ]

    if is_train_set:
        positive_samples = list(filter(lambda ctx: ctx.has_answer, ctxs))
        negative_samples = list(filter(lambda ctx: not ctx.has_answer, ctxs))
    else:
        positive_samples = []
        negative_samples = ctxs

    positive_ctxs_from_gold_page = (
        list(
            filter(
                lambda ctx: _is_from_gold_wiki_page(
                    gold_passage_map, ctx.title, question
                ),
                positive_samples,
            )
        )
        if gold_page_only_positives
        else []
    )

    def find_answer_spans(ctx: ReaderPassage):
        # for retriever results passages which have already been processed
        if ctx.closest_span is not None:
            assert ctx.has_answer == True
        # for gold info passages
        if ctx.closest_span is None:
            ctx.has_answer, ctx.closest_span = ctx.get_span_from_answers(answers, ctx.passage_text)
        if ctx.has_answer:
            if ctx.passage_token_ids is None:
                ctx.passage_token_ids = tensorizer.text_to_tensor(
                    ctx.passage_text, add_special_tokens=False
                )
            closest_span_token_ids = tensorizer.text_to_tensor(ctx.closest_span, add_special_tokens=False)
            answer_spans = [
                _find_answer_positions(ctx.passage_token_ids, closest_span_token_ids)
                # for i in range(len(answers))
            ]

            # flatten spans list
            answer_spans = [item for sublist in answer_spans for item in sublist]
            answers_spans = list(filter(None, answer_spans))
            ctx.answers_spans = answers_spans

            if not answers_spans:
                # can happen if passage got cropped due to encoder limit and the cropped portion had the answer
                logger.warning(
                    "No answer found in passage id=%s text=%s, answers=%s, question=%s",
                    ctx.id,
                    ctx.passage_text,
                    answers,
                    question,
                )

            ctx.has_answer = bool(answers_spans)

        return ctx

    # check if any of the selected ctx+ has answer spans
    selected_positive_ctxs = list(
        filter(
            lambda ctx: ctx.has_answer,
            [find_answer_spans(ctx) for ctx in positive_ctxs_from_gold_page],
        )
    )

    if not selected_positive_ctxs:  # fallback to positive ctx not from gold pages
        selected_positive_ctxs = list(
            filter(
                lambda ctx: ctx.has_answer,
                [find_answer_spans(ctx) for ctx in positive_samples],
            )
        )[0:max_positives]

    # optionally include gold passage itself if it is still not in the positives list
    
    if include_gold_passage:
        assert question in gold_passage_map
        gold_passage = gold_passage_map[question]
        included_gold_passage = next(
            iter(ctx for ctx in selected_positive_ctxs if ctx.id == gold_passage.id),
            None,
        )
        if not included_gold_passage:
            gold_passage = find_answer_spans(gold_passage)
            if not gold_passage.has_answer:
                pass
                # removing unecessary verbose warnings
                # logger.warning("No answer found in gold passage %s", gold_passage.title)
            else:
                selected_positive_ctxs.append(gold_passage)

    max_negatives = (
        min(max(10 * len(selected_positive_ctxs), max1_negatives), max2_negatives)
        if is_train_set
        else DEFAULT_EVAL_PASSAGES
    )
    negative_samples = negative_samples[0:max_negatives]
    return selected_positive_ctxs, negative_samples


def _find_answer_positions(ctx_ids: T, answer: T) -> List[Tuple[int, int]]:
    c_len = ctx_ids.size(0)
    a_len = answer.size(0)
    answer_occurences = []
    for i in range(0, c_len - a_len + 1):
        if (answer == ctx_ids[i : i + a_len]).all():
            answer_occurences.append((i, i + a_len - 1))
    return answer_occurences

def _truncated_sequence(t1: T, t2: T, middle_sep: T = None, tailing_sep: T = None, encoder_seq_length: int = 256):

    t1 = t1.tolist()
    t2 = t2.tolist()


    sep_token = 102
    assert sep_token not in t2
    assert middle_sep is None
    assert tailing_sep is None
    passage_threshold = encoder_seq_length // 3
    # truncating logic here
    sep_token_idxs = [i for i, t in enumerate(t1) if t == sep_token]
    assert len(sep_token_idxs) >= 3
    assert t1[-1] == sep_token
    question_passage_separator_idx = sep_token_idxs[-3]
    sep_token_idxs = sep_token_idxs[:-3]
    question = t1[:question_passage_separator_idx]
    passage_title = t1[question_passage_separator_idx:]
    query_sub_parts = []
    start_idx = 0
    for sep_token_idx in sep_token_idxs:
        end_idx = sep_token_idx + 1
        query_sub_parts.append(question[start_idx:end_idx])
        start_idx = end_idx
    query_sub_parts.append(question[start_idx:])
    query_sub_part_lengths = [len(sub_part) for sub_part in query_sub_parts]
    if len(query_sub_parts) == 1:
        query_sub_parts.append([])
    effective_length = len(query_sub_parts[0]) + len(query_sub_parts[-1]) + len(passage_title) + len(t2)
    while effective_length > encoder_seq_length and len(t2) > passage_threshold:
        t2 = t2[:-1]
        effective_length -=1
    query_sub_parts_idxs_to_add = []
    for i in range(len(query_sub_parts) - 2, 0, -1):
        if query_sub_part_lengths[i] + effective_length < encoder_seq_length:
            query_sub_parts_idxs_to_add.append(i)
            effective_length += query_sub_part_lengths[i]
        else:
            break
    t1_token_ids = list(query_sub_parts[0])
    for idx in reversed(query_sub_parts_idxs_to_add):
        t1_token_ids += query_sub_parts[idx]
    t1_token_ids += query_sub_parts[-1]
    t1_token_ids += passage_title
    # makes the assumption that 1st & last question won't exceed encoder_seq_length
    return torch.tensor(t1_token_ids), torch.tensor(t2)


def _concat_pair(t1: T, t2: T, middle_sep: T = None, tailing_sep: T = None, encoder_seq_length: int = 256):
    # t1 - question and title, t2 - passage tokens
    assert middle_sep is None
    middle = [middle_sep] if middle_sep else []
    r = [t1] + middle + [t2] + ([tailing_sep] if tailing_sep else [])
    r = torch.cat(r, dim=0)
    if len(r) <= encoder_seq_length:
        return r, t1.size(0) + len(middle)
    else:
        t1, t2 = _truncated_sequence(t1, t2, middle_sep, tailing_sep, encoder_seq_length)
        r = [t1] + middle + [t2] + ([tailing_sep] if tailing_sep else [])
        r = torch.cat(r, dim=0)
        len(r) <= encoder_seq_length
        return r, t1.size(0) + len(middle)


def _get_gold_ctx_dict(file: str, extra_text: str) -> Tuple[Dict[str, ReaderPassage], Dict[str, str]]:
    gold_passage_infos = (
        {}
    )  # question|question_tokens -> ReaderPassage (with title and gold ctx)

    # original NQ dataset has 2 forms of same question - original, and tokenized.
    # Tokenized form is not fully consisted with the original question if tokenized by some encoder tokenizers
    # Specifically, this is the case for the BERT tokenizer.
    # Depending of which form was used for retriever training and results generation, it may be useful to convert
    # all questions to the canonical original representation.
    original_questions = {}  # question from tokens -> original question (NQ only)

    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)

    for sample in data:
        question = sample["question"].replace("'", "’")
        question_from_tokens = (
            sample["question_tokens"].replace("'", "’") if "question_tokens" in sample else question
        )
        original_questions[question_from_tokens] = question
        assert question == question_from_tokens, "Question and question_tokens are not equal"
        title = sample["title"]
        context = sample["context"]  # Note: This one is cased
        rp = ReaderPassage(sample["example_id"], text=context, title=title, extra_text=extra_text)
        if question in gold_passage_infos:
            pass
            # removing unecessary verbose warnings
            # logger.info("Duplicate question %s", question)
            # rp_exist = gold_passage_infos[question]
            # logger.info(
            #     "Duplicate question gold info: title new =%s | old title=%s",
            #     title,
            #     rp_exist.title,
            # )
            # logger.info("Duplicate question gold info: new ctx =%s ", context)
            # logger.info(
            #     "Duplicate question gold info: old ctx =%s ", rp_exist.passage_text
            # )

        gold_passage_infos[question] = rp
        gold_passage_infos[question_from_tokens] = rp
    return gold_passage_infos, original_questions


def _is_from_gold_wiki_page(
    gold_passage_map: Dict[str, ReaderPassage], passage_title: str, question: str
):
    assert question in gold_passage_map
    gold_info = gold_passage_map.get(question, None)
    # if gold_info:
    return passage_title.lower() == gold_info.title.lower()
    # return False


def _extend_span_to_full_words(
    tensorizer: Tensorizer, tokens: List[int], span: Tuple[int, int]
) -> Tuple[int, int]:
    start_index, end_index = span
    max_len = len(tokens)
    while start_index > 0 and tensorizer.is_sub_word_id(tokens[start_index]):
        start_index -= 1

    while end_index < max_len - 1 and tensorizer.is_sub_word_id(tokens[end_index + 1]):
        end_index += 1

    return start_index, end_index


def _preprocess_reader_samples_chunk(
    samples: List,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    is_train_set: bool,
    encoder_seq_length: int,
) -> str:
    chunk_id, samples = samples
    logger.info("Start batch %d", len(samples))
    iterator = preprocess_retriever_data(
        samples,
        gold_passages_file,
        tensorizer,
        is_train_set=is_train_set,
        encoder_seq_length=encoder_seq_length,
    )

    results = []

    iterator = tqdm(iterator)
    for i, r in enumerate(iterator):
        r.on_serialize()
        results.append(r)

    out_file = out_file_prefix + "." + str(chunk_id) + ".pkl"
    with open(out_file, mode="wb") as f:
        logger.info("Serialize %d results to %s", len(results), out_file)
        pickle.dump(results, f)
    return out_file
