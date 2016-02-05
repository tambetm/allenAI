import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
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
  assert args.batch_size % 3 == 0, "Batch size must be multiple of 3"

  if args.rnn == 'GRU':
    RNN = recurrent.GRU
  elif args.rnn == 'LSTM':
    RNN = recurrent.LSTM
  else:
    assert False, "Invalid RNN"

  if args.bidirectional:
    assert not args.convolution, "Convolutional layer is not supported with bidirectional RNN"
    assert not args.pooling, "Pooling layer is not supported with bidirectional RNN"
    assert args.dense_layers == 0, "Dense layers are not supported with bidirectional RNN"
    model = Graph()
    model.add_input(name="input", batch_input_shape=(args.batch_size,1), dtype="uint")
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
    model.add(Embedding(vocab_size, args.embed_size, mask_zero=not args.convolution))
    if args.convolution:
      model.add(Convolution1D(nb_filter=args.conv_filters,
                          filter_length=args.conv_filter_length,
                          border_mode=args.conv_border_mode,
                          activation=args.conv_activation,
                          subsample_length=args.conv_subsample_length))
      if args.pooling:
        model.add(MaxPooling1D(pool_length=args.pool_length))
    for i in xrange(args.layers):
      model.add(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True))
      if args.dropout > 0:
        model.add(Dropout(args.dropout))
    for i in xrange(args.dense_layers):
      if i + 1 == args.dense_layers:
        model.add(Dense(args.hidden_size, activation='linear'))
      else:
        model.add(Dense(args.hidden_size, activation=args.dense_activation))

  return model

def compile_model(model, args):
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

  def GESD(y_true, y_pred):
      assert y_true.ndim == 2
      assert y_pred.ndim == 2
      y_true = l2_normalize(y_true, axis=1)
      y_pred = l2_normalize(y_pred, axis=1)
      eucledian_dist = T.sqrt(T.sum(T.square(y_true - y_pred), axis=1, keepdims=True))
      part1 = 1.0 / (1.0 + eucledian_dist)
      gamma = 1.0
      c = 1.0
      part2 = 1.0 / (1.0 + T.exp(-gamma * (T.sum(y_true * y_pred, axis=1, keepdims=False) + c)))
      return T.sum(part1 * part2, axis=1, keepdims=False)

  def AESD(y_true, y_pred):
      assert y_true.ndim == 2
      assert y_pred.ndim == 2
      y_true = l2_normalize(y_true, axis=1)
      y_pred = l2_normalize(y_pred, axis=1)
      eucledian_dist = T.sqrt(T.sum(T.square(y_true - y_pred), axis=1, keepdims=True))
      part1 = 1.0 / (1.0 + eucledian_dist)
      gamma = 1.0
      c = 1.0
      part2 = 1.0 / (1.0 + T.exp(-gamma * (T.sum(y_true * y_pred, axis=1, keepdims=False) + c)))
      return T.sum(part1 + part2, axis=1, keepdims=False)

  def GESD_ranking_loss(y_true, y_pred):
      q = y_pred[0::3]
      a_correct = y_pred[1::3]
      a_incorrect = y_pred[2::3]

      return mean(T.maximum(0., args.margin - GESD(q, a_correct) + GESD(q, a_incorrect)) - y_true[0]*0, axis=-1)

  def AESD_ranking_loss(y_true, y_pred):
      q = y_pred[0::3]
      a_correct = y_pred[1::3]
      a_incorrect = y_pred[2::3]

      return mean(T.maximum(0., args.margin - AESD(q, a_correct) + AESD(q, a_incorrect)) - y_true[0]*0, axis=-1)

  if args.loss == 'cosine':
    loss = cosine_ranking_loss
  elif args.loss == 'gesd':
    loss = GESD_ranking_loss
  elif args.loss == 'aesd':
    loss = AESD_ranking_loss
  else:
    assert False, "Unknown loss function"

  if args.bidirectional:
    model.compile(optimizer=args.optimizer, loss={'output': loss})
  else:
    model.compile(optimizer=args.optimizer, loss=loss)

class TestAccuracy(Callback):
  def __init__(self, csv_file, args):
    super(Callback, self).__init__()
    tokenizer = load_tokenizer(args.load_tokenizer)
    ids, q, a, b, c, d, self.corrects = load_test_data(csv_file)
    self.test_data = convert_test_data(q, a, b, c, d, tokenizer, args.maxlen)
    self.args = args

  def on_epoch_end(self, epoch, logs={}):
    pred = predict_data(self.model, self.test_data, self.args)
    preds, sims = convert_test_predictions(pred)
    acc = calculate_accuracy(preds, self.corrects)
    # TODO: how to do this?
    logs['val_acc'] = acc

def default_callbacks(args, callbacks=[]):
  if args.csv_file:
    callbacks.append(TestAccuracy(args.csv_file, args))
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
    return model.fit({'input': data, 'output': np.empty((data.shape[0], args.hidden_size))}, 
        batch_size=args.batch_size, nb_epoch=args.epochs, 
        validation_split=args.validation_split, verbose=args.verbose, callbacks=callbacks,
        shuffle=False)
  else:
    return model.fit(data, np.empty((data.shape[0], args.hidden_size)), batch_size=args.batch_size, nb_epoch=args.epochs,
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
  parser.add_argument("--batch_size", type=int, default=300)
  parser.add_argument("--margin", type=float, default=0.2)
  parser.add_argument("--dense_layers", type=int, default=0)
  parser.add_argument("--dense_activation", choices=['relu','sigmoid','tanh'], default='relu')
  parser.add_argument("--convolution", action='store_true', default=False)
  parser.add_argument("--conv_filters", type=int, default=1000)
  parser.add_argument("--conv_filter_length", type=int, default=3)
  parser.add_argument("--conv_activation", choices=['relu','sigmoid','tanh'], default='relu')
  parser.add_argument("--conv_subsample_length", type=int, default=1)
  parser.add_argument("--conv_border_mode", choices=['valid','same'], default='valid')
  parser.add_argument("--pooling", action='store_true', default=False)
  parser.add_argument("--pool_length", type=int, default=2)
  parser.add_argument("--loss", choices=['cosine', 'gesd', 'aesd'], default='cosine')

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
