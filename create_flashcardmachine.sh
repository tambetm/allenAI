#!/bin/bash

python src/allen_preproc_simple.py --data /storage/hpc_tanel/allenAI/X_flashcardmachine_qa_cleaner_6_to_10_no_qm_ranking_shuffled.txt \
  --tokenizer_save_path model/tokenizer_flashcardmachine_full.pkl data/flashcardmachine_maxlen100.pkl --maxlen 100
