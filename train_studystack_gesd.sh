#!/bin/bash

python src/allen_fit_simple.py --data data/studystack_full.pkl model/studystack_gsde --loss gesd $*
