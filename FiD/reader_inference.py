import torch
import transformers
import numpy as np
from pathlib import Path
from time import time

from FiD.src.data import MinimalCollator
from FiD.src.data import format_conversational_example
from FiD.src.model import FiDT5

ENCODER_MAXLENGTH = 512
ANSWER_MAXLENGTH = 100
MODEL_PATH = '/home/toolkit/topiocqa/downloads/model_checkpoints/reader/fid/dpr_retriever/all_history/checkpoint/best_dev'
NUM_PASSAGES = 50

import logging
logger = logging.getLogger(__name__)

class ReaderForInference():

    def __init__(self):

        self.tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        self.collator = MinimalCollator(ENCODER_MAXLENGTH, self.tokenizer)
        model_class = FiDT5
        self.model = model_class.from_pretrained(MODEL_PATH)
        self.model.cuda()

    def get_answer(self, question, passages):
        context_ids, context_mask = self.form_input(question, passages)
        time_start = time()
        outputs = self.model.generate(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            max_length=ANSWER_MAXLENGTH,
        )

        output = outputs[0]
        ans = self.tokenizer.decode(output, skip_special_tokens=True)
        # logger.info("Time to generate answer: %.2f", time() - time_start)
        # logger.info("Answer: %s", ans)
        return ans

    def form_input(self,
                   question,
                   passages,
                   title_prefix='title:',
                   sub_title_prefix='sub-title:',
                   passage_prefix='context:'):
        
        input = {
            'question': question,
            'ctxs': passages,
        }
        truncated_input = format_conversational_example(input,
                                              title_prefix=title_prefix,
                                              sub_title_prefix=sub_title_prefix,
                                              passage_prefix=passage_prefix,
                                              tokenizer=self.tokenizer,
                                              max_length=ENCODER_MAXLENGTH)
        truncated_input = {
            'passages': [c['text'] for c in truncated_input['ctxs'][:NUM_PASSAGES]]
        }

        passage_ids, passage_masks = self.collator([truncated_input])

        return passage_ids, passage_masks
