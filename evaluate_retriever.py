import json
from argparse import ArgumentParser


def hits_at_n(ranks, n):
    if len(ranks) == 0:
        return 0
    else:
        return len([x for x in ranks if x <= n]) * 100.0 / len(ranks)

def main(data_file, results_file):

    final_scores = {}
    with open(data_file, 'r') as f:
        data = json.load(f)

    with open(results_file, 'r') as f:
        results = json.load(f)

    ranks = []

    for i, sample in enumerate(data):
        question = sample["question"]
        gold_ctx = sample["positive_ctxs"][0]
        rank_assigned = False
        for rank, ctx in enumerate(results[i]["ctxs"]):
            if ctx["title"] == gold_ctx["title"] and ctx["text"] == gold_ctx["text"]:
                ranks.append(float(rank + 1))
                rank_assigned = True
                break
        if not rank_assigned:
            ranks.append(1000.0)


    for n in [1, 3, 5, 10, 20, 30 ,50, 100]:
        score = hits_at_n(ranks, n)
        final_scores["Hits@" + str(n)] = score

    print(json.dumps(final_scores, indent=4, sort_keys=True))

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default='downloads/data/retriever/all_history/dev.json')
    parser.add_argument("--results_file", type=str, default='downloads/results/retriever/dpr/all_history/dev.json')
    args = parser.parse_args()
    main(args.data_file, args.results_file)
