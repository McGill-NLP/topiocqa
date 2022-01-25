import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

WIKI_FILE = "downloads/data/wikipedia_split/full_wiki_segments.tsv"
OUTPUT_FILE = "downloads/data/wikipedia_split/bm25_collection/full_wiki_segments_pyserini_format.jsonl"

id_col= 0
text_col= 1
title_col = 2

def main(wiki_file, output_file):

    with open(wiki_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        with open(output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):
                if row[id_col] == "id":
                    continue
                title = row[title_col]
                text = row[text_col]
                title = ' '.join(title.split(' [SEP] '))
                obj = {"contents": " ".join([title, text]), "id": f"doc{i}"}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--wiki_file", type=str, default=WIKI_FILE)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    main(args.wiki_file, args.output_file)

