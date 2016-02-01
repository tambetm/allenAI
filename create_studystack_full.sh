#!/bin/bash

python src/allen_preproc_simple.py --data /storage/hpc_tanel/allenAI/X_studystack_qa_cleaner_no_qm_ranking_shuffled.txt \
  --tokenizer_save_path model/tokenizer_studystack_full.pkl data/studystack_full.pkl
