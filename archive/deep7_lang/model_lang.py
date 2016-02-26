import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers import recurrent
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Graph, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from keras.models import model_from_json
import theano.tensor as T
from theano import function
from itertools import islice
from predict import *

def create_model(vocab_size, args):
  if args.rnn == 'GRU':
    RNN = recurrent.GRU
  elif args.rnn == 'LSTM':
    RNN = recurrent.LSTM
  else:
    assert False, "Invalid RNN"

  if args.bidirectional:
    model = Graph()
    model.add_input(name="input", batch_input_shape=(args.batch_size,1), dtype="uint")
    model.add_node(Embedding(vocab_size, args.embed_size, mask_zero=True), name="embed", input='input')
    for i in xrange(args.layers):
      model.add_node(RNN(args.hidden_size, return_sequences=True), 
          name='forward'+str(i+1), 
          input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
          inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
      model.add_node(RNN(args.hidden_size, return_sequences=True, go_backwards=True), 
          name='backward'+str(i+1), 
          input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
          inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
      if args.dropout > 0:
        model.add_node(Dropout(args.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])
    model.add_node(TimeDistributedDense(vocab_size, activation="softmax"), name="softmax",
        input='dropout'+str(args.layers) if args.dropout > 0 else None,
        inputs=['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else [])
    model.add_output(name='output', input="softmax")
  else:
    model = Sequential()
    model.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
    for i in xrange(args.layers):
      model.add(RNN(args.hidden_size, return_sequences=True))
      if args.dropout > 0:
        model.add(Dropout(args.dropout))
    model.add(TimeDistributedDense(vocab_size, activation="softmax"))

  return model

def compile_model(model, args):
  if args.bidirectional:
    model.compile(optimizer=args.optimizer, loss={'output': 'categorical_crossentropy'})
  else:
    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy')

class TestAccuracy(Callback):
  def __init__(self, csv_file, args):
    super(Callback, self).__init__()
    tokenizer = load_tokenizer(args.load_tokenizer)
    ids, q, a, b, c, d, self.corrects = load_test_data(csv_file)
    self.test_data = convert_test_data(q, a, b, c, d, tokenizer, args.maxlen)
    self.args = args

  def on_epoch_end(self, epoch, logs={}):
    pred = predict_data(self.model, self.test_data, self.args)
    preds = convert_test_predictions(pred)
    acc = calculate_accuracy(preds, self.corrects)
    # TODO: how to do this?
    logs['val_acc'] = acc

def default_callbacks(args, callbacks=[]):
  #if args.csv_file:
  #  callbacks.append(TestAccuracy(args.csv_file, args))
  if args.model_path:
    callbacks.append(ModelCheckpoint(filepath="%s_{epoch:02d}_acc_{val_acc:.4f}.hdf5" % args.model_path, monitor="val_acc", verbose=1, save_best_only=False))
  if args.patience:
    callbacks.append(EarlyStopping(patience=args.patience, monitor="val_acc", verbose=1))
  if args.lr_epochs:
    def lr_scheduler(epoch):
      lr = args.lr / 2**int(epoch / args.lr_epochs)
      print "Epoch %d: learning rate %g" % (epoch + 1, lr)
      return lr
    callbacks.append(LearningRateScheduler(lr_scheduler))
  return callbacks

def fit_data(model, data, args):
  callbacks = default_callbacks(args)

  if args.bidirectional:
    return model.fit({'input': data[0], 'output': data[1]}, 
        batch_size=args.batch_size, nb_epoch=args.epochs, 
        validation_split=args.validation_split, verbose=args.verbose, callbacks=callbacks,
        shuffle=False)
  else:
    return model.fit(data[0], data[1], batch_size=args.batch_size, nb_epoch=args.epochs,
        validation_split=args.validation_split, verbose=args.verbose, callbacks=callbacks,
        shuffle=False)

def fit_generator(model, generator, args, callbacks = []):
  callbacks = default_callbacks(args)

  if args.bidirectional:
    return model.fit_generator(generator, samples_per_epoch=args.samples_per_epoch,
        nb_epoch=args.epochs, verbose=args.verbose, callbacks=callbacks)
  else:
    return model.fit_generator(generator, samples_per_epoch=args.samples_per_epoch,
        nb_epoch=args.epochs, verbose=args.verbose, callbacks=callbacks)

def predict_data(model, data, args):
  if args.bidirectional:
    pred = model.predict({'input': data}, batch_size=args.batch_size, verbose=args.verbose)
    pred = pred['output']
  else:
    pred = model.predict(data, batch_size=args.batch_size, verbose=args.verbose)

  #print "Predictions: ", pred.shape
  return pred

def add_model_params(parser):
  parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
  parser.add_argument("--embed_size", type=int, default=300)
  parser.add_argument("--hidden_size", type=int, default=1024)
  parser.add_argument("--layers", type=int, default=1)
  parser.add_argument("--dropout", type=float, default=0)
  parser.add_argument("--bidirectional", action='store_true', default=False)
  parser.add_argument("--batch_size", type=int, default=100)
  parser.add_argument("--margin", type=float, default=0.1)

def add_training_params(parser):
  parser.add_argument("--validation_split", type=float, default=0)
  parser.add_argument("--optimizer", choices=['adam', 'rmsprop', 'sgd'], default='adam')
  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--lr_epochs", type=int, default=0)
  parser.add_argument("--samples_per_epoch", type=int, default=1500000)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
