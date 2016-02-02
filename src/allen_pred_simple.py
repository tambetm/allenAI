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
    q = y_pred[0::3]
    a_correct = y_pred[1::3]
    a_incorrect = y_pred[2::3]

    return mean(T.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true[0]*0, axis=-1)

def np_l2_normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / norm

def np_cosine_similarity(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    y_true = np_l2_normalize(y_true, axis=1)
    y_pred = np_l2_normalize(y_pred, axis=1)
    return np.sum(y_true * y_pred, axis=1, keepdims=False)

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--csv_file", default="data/training_set.tsv")
parser.add_argument("--write_predictions")
parser.add_argument("--tokenizer", default="model/tokenizer_studystack_full.pkl")
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
parser.add_argument("--margin", type=float, default=0.01)
parser.add_argument("--dense_layers", type=int, default=0)
parser.add_argument("--dense_activation", choices=['relu','sigmoid','tanh'], default='relu')
args = parser.parse_args()

print "Loading data..."
ids = []
questions = []
corrects = []
answersA = []
answersB = []
answersC = []
answersD = []
with open(args.csv_file) as f:
  reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
  line = next(reader)  # ignore header
  is_train_set = (len(line) == 7)
  for line in reader:
    ids.append(line[0])
    questions.append(line[1])
    if is_train_set:
      corrects.append(line[2])
      answersA.append(line[3])
      answersB.append(line[4])
      answersC.append(line[5])
      answersD.append(line[6])
    else:
      answersA.append(line[2])
      answersB.append(line[3])
      answersC.append(line[4])
      answersD.append(line[5])
print "Questions: ", len(questions)
assert len(questions) == len(answersA) == len(answersB) == len(answersC) == len(answersD)
assert not is_train_set or len(corrects) == len(questions)

print "Sample question and answers:"
for i in xrange(3):
  print questions[i], "A:", answersA[i], "B:", answersB[i], "C:", answersC[i], "D:", answersD[i], "Correct: ", corrects[i] if is_train_set else '?'

texts = questions + answersA + answersB + answersC + answersD
print "Texts size:", len(texts)

tokenizer = pickle.load(open(args.tokenizer, "rb"))
sequences = tokenizer.texts_to_sequences(texts)

if args.maxlen:
  maxlen = args.maxlen
else:
  maxlen = max([len(s) for s in sequences])
print "Sequences maxlen:", maxlen

texts = pad_sequences(sequences, maxlen=maxlen) 

vocab_size = tokenizer.nb_words if tokenizer.nb_words else len(tokenizer.word_index)+1
if args.vocab_size:
  print "Overriding original vocabulary size", vocab_size
  vocab_size = args.vocab_size
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
  for i in xrange(args.dense_layers):
    if i + 1 == args.dense_layers:
      model.add(Dense(args.hidden_size, activation='linear'))
    else:
      model.add(Dense(args.hidden_size, activation=args.dense_activation))

model.summary()

print "Loading weights from %s" % args.model_path
model.load_weights(args.model_path)

print "Compiling model..."
if args.bidirectional:
  model.compile(optimizer=args.optimizer, loss={'output': cosine_ranking_loss})
else:
  model.compile(optimizer=args.optimizer, loss=cosine_ranking_loss)

if args.bidirectional:
  pred = model.predict({'input': texts}, batch_size=args.batch_size, verbose=args.verbose)
  pred = pred['output']
else:
  pred = model.predict(texts, batch_size=args.batch_size, verbose=args.verbose)

print "Predictions: ", pred.shape

qlen = int(pred.shape[0] / 5)
questions = pred[0:qlen]
answersA = pred[qlen:2*qlen]
answersB = pred[2*qlen:3*qlen]
answersC = pred[3*qlen:4*qlen]
answersD = pred[4*qlen:5*qlen]
print "Predicted vectors:", questions.shape, answersA.shape, answersB.shape, answersC.shape, answersD.shape

print "Question:", questions[0,:4]
print "Answer A:", answersA[0,:4]
print "Answer B:", answersB[0,:4]
print "Answer C:", answersC[0,:4]
print "Answer D:", answersD[0,:4]

sims = np.array([
  np_cosine_similarity(questions, answersA),
  np_cosine_similarity(questions, answersB),
  np_cosine_similarity(questions, answersC),
  np_cosine_similarity(questions, answersD)
])
print "Similarities:", sims.shape
print "Question 1:", sims[:,0]
print "Question 2:", sims[:,1]
print "Question 3:", sims[:,2]

preds = np.argmax(sims, axis=0)
print "Predictions:", preds.shape
print "Predicted answers:", preds[:3]

preds = [chr(ord('A') + p) for p in preds]
print "Predicted answers:", preds[:3]

if args.write_predictions:
  print "Writing predictions to", args.write_predictions
  f = open(args.write_predictions, "w")
  f.write("id,correctAnswer")
  for i in xrange(len(preds)):
    f.write("%s,%s" % (ids[i], preds[i]))
  f.close()

if is_train_set:
  correct = sum([corrects[i] == p for i,p in enumerate(preds)])
  print "Correct: %d Total: %d Accuracy: %f" % (correct, len(preds), float(correct) / len(preds))
