import csv
import json
import os
from argparse import ArgumentParser

from pyserini.search import SimpleSearcher
from tqdm import tqdm

WIKI_FILE = "downloads/data/wikipedia_split/full_wiki_segments.tsv"
BM25_INDEX_PATH = "downloads/data/wikipedia_split/indexes/bm25"

id_col= 0
text_col= 1
title_col = 2


def main(input_file, output_file, wiki_file, bm25_index_path):
    corpus = []
    with open(wiki_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[id_col] == "id":
                continue
            title = row[title_col]
            text = row[text_col]
            
            doc_title = title.split(' [SEP] ')[0]
            doc_sub_title = title.split(' [SEP] ')[1]
            doc_contents = text
            corpus.append({"title": doc_title, "sub_title": doc_sub_title, "contents": doc_contents})

    searcher = SimpleSearcher(BM25_INDEX_PATH)
    searcher.set_bm25(0.9, 0.4)

    print('Processing', input_file)
    with open(input_file, 'r', encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    for query in tqdm(queries):
        obj = {}
        obj['question'] = query['question']
        obj['answers'] = query['answers']
        obj["conv_id"] = str(query["conv_id"])
        obj["turn_id"] = str(query["turn_id"])
        if '[SEP]' in query['question']:
            question = query['question'].replace('[SEP] ', '')
        else:
            question = query['question']
        hits = searcher.search(question, k=100)
        obj['ctxs'] = []
        for hit in hits:
            doc_idx = int(hit.docid.strip('doc')) - 1
            ctx = {}
            ctx['id'] = f"wiki:{doc_idx + 1}"
            ctx['score'] = hit.score
            ctx['has_answer'] = False
            ctx["title"] = corpus[doc_idx]["title"] + " [SEP] " + corpus[doc_idx]["sub_title"]
            ctx["text"] = corpus[doc_idx]["contents"]
            obj['ctxs'].append(ctx)
        results.append(obj)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as output:
        output.write(json.dumps(results, indent=4) + "\n")



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--input_file', type=str, help='Path to input file',
        default='downloads/data/retriever/all_history/dev.json')
    parser.add_argument('--output_file', type=str, help='Path to output results file',
        default='downloads/results/retriever/bm25/all_history/dev.json')
    parser.add_argument("--wiki_file", type=str, default=WIKI_FILE)
    parser.add_argument("--bm25_index_path", type=str, default=BM25_INDEX_PATH)
    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.wiki_file, args.bm25_index_path)
