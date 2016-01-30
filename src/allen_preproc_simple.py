import argparse
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("--data", default="/storage/allenAI/X_studystack_qa_cleaner_ranking.txt")
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
tokenizer = Tokenizer(args.max_words)
tokenizer.fit_on_texts(lines)
print "Number of words: ", len(tokenizer.word_index)

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

pickle.dump(sequences, open("sequences.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

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
pickle.dump(sequences_padded, open(args.save_path, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"
