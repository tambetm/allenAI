import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import base_filter
import cPickle as pickle

def fit_tokenizer(lines, max_words = None):
  tokenizer = Tokenizer(max_words, filters=base_filter().replace('_', ''))
  tokenizer.fit_on_texts(lines)
  return tokenizer

def load_tokenizer(tokenizer_path):
  return pickle.load(open(tokenizer_path, "rb"))

def save_tokenizer(tokenizer_path):
  pickle.dump(tokenizer, open(tokenizer_path, "wb"), pickle.HIGHEST_PROTOCOL)

def vocabulary_size(tokenizer):
  return tokenizer.nb_words+1 if tokenizer.nb_words else len(tokenizer.word_index)+1

def text_to_data(lines, tokenizer, maxlen):
  sequences = tokenizer.texts_to_sequences(lines)
  # apply maxlen limitation only when sequences are longer
  seqmaxlen = max([len(s) for s in sequences])
  if seqmaxlen > maxlen:
    seqmaxlen = maxlen
  data = pad_sequences(sequences, maxlen=seqmaxlen) 
  return data

def add_data_params(parser):
  parser.add_argument("--max_words", type=int)
  parser.add_argument("--maxlen", type=int)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("text_path")
  parser.add_argument("--save_data")
  parser.add_argument("--load_tokenizer")
  parser.add_argument("--save_tokenizer")
  parser.add_argument("--save_sequences")
  add_data_params(parser)
  args = parser.parse_args()

  print "Loading data..."
  with open(args.text_path) as f:
    lines = list(f)

  print "Lines:", len(lines)

  print "Sample question, correct answer, incorrect answer:"
  for i in xrange(3):
      print lines[i]

  if args.load_tokenizer:
    print "Loading tokenizer", args.load_tokenizer
    tokenizer = load_tokenizer(args.load_tokenizer)
  else:
    print "Tokenizing data..."
    tokenizer = fit_tokenizer(lines, args.max_words)

  print "Number of words: ", len(tokenizer.word_index)
  wcounts = tokenizer.word_counts.items()
  wcounts.sort(key=lambda x: x[1], reverse=True)
  print "Most frequent words:", wcounts[:10]
  print "Most rare words:", wcounts[-10:]
  print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

  if args.save_tokenizer:
    print "Saving tokenizer to", args.save_tokenizer
    save_tokenizer(args.save_tokenizer)

  if args.save_data:
    print "Converting text to sequences..."
    sequences = tokenizer.texts_to_sequences(lines)

    print "Sample sequences:"
    for i in xrange(3):
      print sequences[i]

    if args.save_sequences:
      print "Saving sequences to", args.save_sequences
      pickle.dump(sequences, open(args.save_sequences, "wb"), pickle.HIGHEST_PROTOCOL)

    print "Padding sequences..."
    sequences_padded = pad_sequences(sequences, maxlen=args.maxlen) 
    print "Sequences padded:", sequences_padded.shape

    print "Sample padded sequences:"
    for i in xrange(3):
      print sequences_padded[i]

    print "Saving data to", args.save_data
    pickle.dump(sequences_padded, open(args.save_data, "wb"), pickle.HIGHEST_PROTOCOL)

  print "Done"
