python drivers/gen_passage_embeddings.py  \
--data_dir=dataset/cast_shared/tokenized  \
--checkpoint=checkpoint/ad-hoc-ance-msmarco  \
--output_dir=dataset/cast_shared/embeddings \
--model_type=rdot_nll