import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import recurrent
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Graph, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import theano.tensor as T
from theano import function
from itertools import islice
from predict2 import *

def create_model(vocab_size, args):
  if args.rnn == 'GRU':
    RNN = recurrent.GRU
  elif args.rnn == 'LSTM':
    RNN = recurrent.LSTM
  else:
    assert False, "Invalid RNN"

  model = Graph()
  model.add_input(name="question", batch_input_shape=(args.batch_size,args.maxlen), dtype="uint")
  model.add_input(name="answer", batch_input_shape=(args.batch_size,args.maxlen), dtype="uint")
  model.add_shared_node(Embedding(vocab_size, args.embed_size, mask_zero=True, input_length=args.maxlen), 
      name="embed", inputs=['question', 'answer'],
      outputs=['question_embed', 'answer_embed'], merge_mode=None)
  model.add_node(RNN(args.hidden_size, return_sequences=False), name="question_rnn", input="question_embed")
  model.add_node(RepeatVector(args.maxlen), name="question_rnn_repeat", input="question_rnn")
  model.add_node(RNN(args.hidden_size, return_sequences=True), name="answer_rnn", 
      inputs=["question_rnn_repeat", "answer_embed"], merge_mode="concat")
  model.add_node(Activation("softmax"), name="output", input="answer_rnn", create_output=True)

  return model

def compile_model(model, args):
  model.compile(optimizer=args.optimizer, loss={'output': 'categorical_crossentropy'})

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

def fit_data(model, data, vocab_size, args):
  callbacks = default_callbacks(args)

  questions = data[0]
  answers = data[1][:,:-1]
  targets = data[1][:,1:]
  print "targets1:", targets.shape
  targets = targets.reshape((-1))
  print "targets2:", targets.shape
  targets = to_categorical(targets, vocab_size)
  print "targets3:", targets.shape
  targets = targets.reshape((-1, args.maxlen, voca_size))
  print "targets4:", targets.shape

  return model.fit({'question': questions, 'answer': answers, 'output': targets}, 
      batch_size=args.batch_size, nb_epoch=args.epochs, 
      validation_split=args.validation_split, verbose=args.verbose, callbacks=callbacks,
      shuffle=False)

def fit_generator(model, generator, args, callbacks = []):
  callbacks = default_callbacks(args)

  return model.fit_generator(generator, samples_per_epoch=args.samples_per_epoch,
        nb_epoch=args.epochs, verbose=args.verbose, callbacks=callbacks)

def predict_data(model, data, args):
  pred1 = model.predict({'question': data[0], 'pos_answer': data[1], 'neg_answer': data[2]}, batch_size=args.batch_size, verbose=args.verbose)
  pred1 = pred1['output']
  #print "Predictions1: ", pred1.shape

  pred2 = model.predict({'question': data[0], 'pos_answer': data[3], 'neg_answer': data[4]}, batch_size=args.batch_size, verbose=args.verbose)
  pred2 = pred2['output']
  #print "Predictions2: ", pred2.shape

  pred = np.concatenate((pred1, pred2[:,args.hidden_size:]), axis=1)
  print "Predictions: ", pred.shape
  return pred

def add_model_params(parser):
  parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
  parser.add_argument("--embed_size", type=int, default=300)
  parser.add_argument("--hidden_size", type=int, default=1024)
  #parser.add_argument("--layers", type=int, default=1)
  #parser.add_argument("--dropout", type=float, default=0)
  #parser.add_argument("--bidirectional", action='store_true', default=False)
  parser.add_argument("--batch_size", type=int, default=100)
  parser.add_argument("--margin", type=float, default=0.2)
  parser.add_argument("--loss", choices=['cosine', 'gesd', 'aesd'], default='cosine')

def add_training_params(parser):
  parser.add_argument("--validation_split", type=float, default=0)
  parser.add_argument("--optimizer", choices=['adam', 'rmsprop', 'sgd'], default='adam')
  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--lr_epochs", type=int, default=0)
  parser.add_argument("--samples_per_epoch", type=int, default=500000)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
