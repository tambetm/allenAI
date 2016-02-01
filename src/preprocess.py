import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("text_path")
parser.add_argument("data_path")
parser.add_argument("--load_tokenizer")
parser.add_argument("--save_tokenizer")
parser.add_argument("--save_sequences")
parser.add_argument("--max_words", type=int)
parser.add_argument("--maxlen", type=int)
args = parser.parse_args()

print "Loading data..."
with open(args.text_path) as f:
  lines = f.readlines()

print "Lines:", len(lines)

print "Sample question, correct answer, incorrect answer:"
for i in xrange(3):
    print lines[i]

if args.load_tokenizer:
  print "Loading tokenizer", args.load_tokenizer
  tokenizer = pickle.load(open(args.load_tokenizer, "rb"))
else:
  print "Tokenizing data..."
  tokenizer = Tokenizer(args.max_words)
  tokenizer.fit_on_texts(lines)

print "Number of words: ", len(tokenizer.word_index)
wcounts = tokenizer.word_counts.items()
wcounts.sort(key=lambda x: x[1], reverse=True)
print "Most frequent words:", wcounts[:10]
print "Most rare words:", wcounts[-10:]
print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

if args.save_tokenizer:
  print "Saving tokenizer to %s..." % args.save_tokenizer
  pickle.dump(tokenizer, open(args.save_tokenizer, "wb"), pickle.HIGHEST_PROTOCOL)

print "Converting text to sequences..."
sequences = tokenizer.texts_to_sequences(lines)

print "Sample sequences:"
for i in xrange(3):
  print sequences[i]

if args.save_sequences:
  print "Saving sequences to %s..." % args.save_sequences
  pickle.dump(sequences, open(args.save_sequences, "wb"), pickle.HIGHEST_PROTOCOL)

print "Padding sequences..."
sequences_padded = pad_sequences(sequences, maxlen=args.maxlen) 
print "Sequences padded:", sequences_padded.shape

print "Sample padded sequences:"
for i in xrange(3):
  print sequences_padded[i]

print "Saving results..."
pickle.dump(sequences_padded, open(args.data_path, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"
