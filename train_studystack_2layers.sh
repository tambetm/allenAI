#!/bin/bash

python src/allen_fit_simple.py --data data/studystack_full.pkl model/studystack_2layers --layers 2 --hidden_size 512 $*
