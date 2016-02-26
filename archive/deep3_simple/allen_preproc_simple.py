import argparse
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import base_filter
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument("--tokenizer_save_path")
parser.add_argument("--sequences_save_path")
parser.add_argument("--data", default="/storage/hpc_tanel/allenAI/X_studystack_qa_cleaner_ranking_shuffled.txt")
parser.add_argument("--max_words", type=int)
parser.add_argument("--maxlen", type=int)
args = parser.parse_args()

print "Loading data..."
with open(args.data) as f:
  lines = f.readlines()

print "Lines:", len(lines)

print "Sample question, correct answer, incorrect answer:"
for i in xrange(3):
    print lines[i]

print "Tokenizing data..."
tokenizer = Tokenizer(args.max_words, filters=base_filter().replace('_', ''))
tokenizer.fit_on_texts(lines)
print "Number of words: ", len(tokenizer.word_index)

if args.tokenizer_save_path:
  print "Saving tokenizer to %s..." % args.tokenizer_save_path
  pickle.dump(tokenizer, open(args.tokenizer_save_path, "wb"), pickle.HIGHEST_PROTOCOL)

wcounts = tokenizer.word_counts.items()
wcounts.sort(key=lambda x: x[1], reverse=True)
print "Most frequent words:", wcounts[:10]
print "Most rare words:", wcounts[-10:]

print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

print "Converting text to sequences..."
sequences = tokenizer.texts_to_sequences(lines)

print "Sample sequences:"
for i in xrange(3):
  print sequences[i]

if args.sequences_save_path:
  print "Saving sequences to %s..." % args.sequences_save_path
  pickle.dump(sequences, open(args.sequences_save_path, "wb"), pickle.HIGHEST_PROTOCOL)

if args.maxlen:
  maxlen = args.maxlen
else:
  maxlen = max([len(s) for s in sequences])
print "Sequences maxlen:", maxlen

print "Padding sequences..."
sequences_padded = pad_sequences(sequences, maxlen=maxlen) 

print "Sample padded sequences:"
for i in xrange(3):
  print sequences_padded[i]

print "Saving results..."
pickle.dump(sequences_padded, open(args.data_path, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"
