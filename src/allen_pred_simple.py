import argparse
import csv
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import recurrent
from keras.models import Graph, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle
import theano.tensor as T
from theano import function

def mean(x, axis=None, keepdims=False):
    return T.mean(x, axis=axis, keepdims=keepdims)

def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / norm

def cosine_similarity(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    y_true = l2_normalize(y_true, axis=1)
    y_pred = l2_normalize(y_pred, axis=1)
    return T.sum(y_true * y_pred, axis=1, keepdims=False)

def cosine_ranking_loss(y_true, y_pred):
    MARGIN = 0.01
    
    q = y_pred[0::3]
    a_correct = y_pred[1::3]
    a_incorrect = y_pred[2::3]

    return mean(T.maximum(0., MARGIN - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true[0]*0, axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--train_csv", default="data/training_set.tsv")
parser.add_argument("--valid_csv", default="data/validation_set.tsv")
parser.add_argument("--tokenizer", default="model/tokenizer.pkl")
parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--bidirectional", action='store_true', default=False)
parser.add_argument("--batch_size", type=int, default=300)
parser.add_argument("--maxlen", type=int)
parser.add_argument("--vocab_size", type=int)
parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam')
parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()

print "Loading data..."
questions = []
corrects = []
answersA = []
answersB = []
answersC = []
answersD = []
with open(args.train_csv) as f:
  reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
  next(reader)  # ignore header
  for line in reader:
    questions.append(line[1])
    corrects.append(line[2])
    answersA.append(line[3])
    answersB.append(line[4])
    answersC.append(line[5])
    answersD.append(line[6])
print "Questions: ", len(questions), "Answers: ", len(corrects)
assert len(questions) == len(corrects) == len(answersA) == len(answersB) == len(answersC) == len(answersD)

print "Sample question and answers:"
for i in xrange(3):
  print questions[i], "A:", answersA[i], "B:", answersB[i], "C:", answersC[i], "D:", answersD[i], "Correct: ", corrects[i]

texts = questions + answersA + answersB + answersC + answersD

tokenizer = pickle.load(open(args.tokenizer, "rb"))
sequences = tokenizer.texts_to_sequences(texts)

if args.maxlen:
  maxlen = args.maxlen
else:
  maxlen = max([len(s) for s in sequences])
print "Sequences maxlen:", maxlen

texts = pad_sequences(sequences, maxlen=maxlen) 

vocab_size = np.max(texts) + 1
if args.vocab_size:
  print "Overriding original vocabulary size", vocab_size
  texts = np.minimum(texts, args.vocab_size - 1)
  vocab_size = np.max(texts) + 1
print "Vocabulary size:", vocab_size, "Texts: ", texts.shape

if args.rnn == 'GRU':
  RNN = recurrent.GRU
elif args.rnn == 'LSTM':
  RNN = recurrent.LSTM
else:
  assert False, "Invalid RNN"

print "Creating model..."

if args.bidirectional:
  model = Graph()
  model.add_input(name="input", batch_input_shape=(args.batch_size,)+texts.shape[1:], dtype="uint")
  model.add_node(Embedding(vocab_size, args.embed_size, mask_zero=True), name="embed", input='input')
  for i in xrange(args.layers):
    model.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True), 
        name='forward'+str(i+1), 
        input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    model.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True, go_backwards=True), 
        name='backward'+str(i+1), 
        input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    if args.dropout > 0:
      model.add_node(Dropout(args.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])
  model.add_output(name='output',
      input='dropout'+str(args.layers) if args.dropout > 0 else None,
      inputs=['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else [])
else:
  model = Sequential()
  model.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
  for i in xrange(args.layers):
    model.add(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True))
    if args.dropout > 0:
      model.add(Dropout(args.dropout))

model.summary()
model.load_weights(args.model_path)

print "Compiling model..."
if args.bidirectional:
  model.compile(optimizer=args.optimizer, loss={'output': cosine_ranking_loss})
else:
  model.compile(optimizer=args.optimizer, loss=cosine_ranking_loss)

if args.bidirectional:
  pred = model.predict({'input': texts}, batch_size=args.batch_size, verbose=args.verbose)
else:
  pred = model.predict(texts, batch_size=args.batch_size, verbose=args.verbose)

print "Predictions: ", pred.shape
