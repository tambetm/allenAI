import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import recurrent
from keras.models import Graph, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import model_from_json
import theano.tensor as T
from theano import function
from itertools import islice

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
    q = y_pred[0::3]
    a_correct = y_pred[1::3]
    a_incorrect = y_pred[2::3]

    return mean(T.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true[0]*0, axis=-1)

def generate_sequences(data, bidirectional):
  y = np.empty((args.batch_size, args.hidden_size))
  while True:
    for i in xrange(0, data.shape[0], args.batch_size):
      #print "Iteration:", i
      x = data[i:i+args.batch_size]
      #print "X,y:", x.shape, y.shape
      if bidirectional:
        yield {'input': x, 'output': y}
      else:
        yield x, y

parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
parser.add_argument("--save_history")
parser.add_argument("--load_model")
parser.add_argument("--save_model")
parser.add_argument("--data_path", default="data/studystack.pkl")
parser.add_argument("--tokenizer_path", default="model/tokenizer_studystack_full.pkl")
parser.add_argument("--maxlen", type=int)
parser.add_argument("--nsamples", type=int)
parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--bidirectional", action='store_true', default=False)
parser.add_argument("--batch_size", type=int, default=300)
parser.add_argument("--samples_per_epoch", type=int, default=1000000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--validation_split", type=float, default=0)
parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam')
#parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--dense_layers", type=int, default=0)
parser.add_argument("--dense_activation", choices=['relu','sigmoid','tanh'], default='relu')
args = parser.parse_args()

assert args.batch_size % 3 == 0, "Batch size must be multiple of 3"

print "Loading data..."
data = pickle.load(open(args.data_path, "rb"))
print "Data:", data.shape
assert data.shape[0] % 3 == 0

if args.nsamples is not None:
  data = data[:args.nsamples*3]

print "Loading tokenizer..."
tokenizer = pickle.load(open(args.tokenizer_path, "rb"))
vocab_size = tokenizer.nb_words+1 if tokenizer.nb_words else len(tokenizer.word_index)+1

if args.load_model:
  model = model_from_json(args.load_model)
else:
  print "Creating model..."

  if args.rnn == 'GRU':
    RNN = recurrent.GRU
  elif args.rnn == 'LSTM':
    RNN = recurrent.LSTM
  else:
    assert False, "Invalid RNN"

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
    assert args.dense_layers == 0, "Bidirectional model doesn't support dense layers yet"
  else:
    model = Sequential()
    model.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
    for i in xrange(args.layers):
      model.add(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True))
      if args.dropout > 0:
        model.add(Dropout(args.dropout))
    for i in xrange(args.dense_layers):
      if i + 1 == args.dense_layers:
        model.add(Dense(args.hidden_size, activation='linear'))
      else:
        model.add(Dense(args.hidden_size, activation=args.dense_activation))

model.summary()

if args.save_model:
  print "Saving model architecture to", args.save_model
  open(args.save_model, 'w').write(model.to_json())

print "Compiling model..."
if args.bidirectional:
  model.compile(optimizer=args.optimizer, loss={'output': cosine_ranking_loss})
else:
  model.compile(optimizer=args.optimizer, loss=cosine_ranking_loss)

callbacks = [ModelCheckpoint(filepath=args.weights_path, verbose=1, save_best_only=False)]
generator = generate_sequences(data, args.bidirectional)

print "Fitting model..."
if args.bidirectional:
  history = model.fit_generator(generator, samples_per_epoch=args.samples_per_epoch,
      nb_epoch=args.epochs, verbose=args.verbose, callbacks=callbacks)
else:
  history = model.fit_generator(generator, samples_per_epoch=args.samples_per_epoch,
      nb_epoch=args.epochs, verbose=args.verbose, callbacks=callbacks)

if args.save_history:
  print "Saving training history to", args.save_history
  pickle.dump(history, open(args.save_history, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"