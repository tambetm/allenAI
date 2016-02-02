#!/bin/bash

python src/allen_fit_simple.py --data data/flashcardmachine_maxlen100.pkl model/flashcardmachine_2048 --hidden_size 2048 $*
