#!/bin/bash

python ../softsegmenter.py \
--hypothesis_file instances.log \
--ref_sentences_file references.txt \
--yaml_file ref_segments.yaml \
--bleu_tokenizer 13a \
--lang en  \
--output_folder segmentation_output