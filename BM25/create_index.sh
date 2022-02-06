OUTPUT=downloads/data/wikipedia_split/indexes/bm25
INPUT=downloads/data/wikipedia_split/bm25_collection

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    python -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 20 \
                            -input ${INPUT} \
                            -index ${OUTPUT}
fi
