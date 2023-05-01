import json
from DPR.dense_retriever_inference import DenseRetrieverForInference
from FiD.reader_inference import ReaderForInference

import hydra
from omegaconf import DictConfig, OmegaConf

import logging

DATASET_FP = '/home/toolkit/topiocqa/topiocqa_dataset/topiocqa_dev.json'

NUM_RESULTS_TO_SHOW = 50
NUM_RESULTS_TO_RETRIEVE = 50


@hydra.main(config_path="DPR/conf", config_name="dense_retriever_inference")
def main(cfg: DictConfig):
    logging.log(logging.INFO, "Loading retriever model...")
    retriever_model = DenseRetrieverForInference(cfg)
    logging.log(logging.INFO, "Loading reader model...")
    reader_model = ReaderForInference()
    logging.log(logging.INFO, "Loading dataset...")

    with open(DATASET_FP, 'r') as f:
        dataset = json.load(f)

    logging.log(logging.INFO, "Evaluating...")
    results = []
    history = []
    for i, data in enumerate(dataset):
        conv_id = data['Conversation_no']
        turn_id = data['Turn_no']
        if turn_id == 1:
            logging.log(logging.INFO, f"Conversation {conv_id}")
            history = []

        question = data['Question']
        history.append(question.strip().strip('?').strip())
        query = ' [SEP] '.join(history)
        retrieved_results = retriever_model.get_top_docs(query, NUM_RESULTS_TO_RETRIEVE)
        passages = [{
            "title": p["title"],
            "text": p["text"],
            "score": float(p["score"]),
        } for p in retrieved_results]

        answer = reader_model.get_answer(query, passages)
        history.append(answer)

        results.append({
            "conv_id": str(conv_id),
            "turn_id": str(turn_id),
            "question": query,
            "predictions": [answer],
        })
    
    logging.log(logging.INFO, "Saving results...")

    with open('/home/toolkit/topiocqa/results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()