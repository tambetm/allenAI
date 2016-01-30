import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import recurrent
from keras.models import Graph, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("--data", default="data/data.pkl")
parser.add_argument("--nsamples", type=int)
parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--bidirectional", action='store_true', default=False)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--validation_split", type=float, default=0.05)
parser.add_argument("--shuffle", choices=['batch', 'true', 'false'], default='true')
parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam')
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()

print "Loading data..."
questions, answers, correct = pickle.load(open(args.data, "rb"))
assert questions.shape[0] == answers.shape[0] == correct.shape[0]

if args.nsamples is not None:
  questions = questions[:args.nsamples]
  answers = answers[:args.nsamples]
  correct = correct[:args.nsamples]

vocab_size = max(np.max(questions), np.max(answers)) + 1
print "Vocabulary size:", vocab_size, "Questions: ", questions.shape, "Answers: ", answers.shape, "Correct: ", correct.shape

if args.rnn == 'GRU':
  RNN = recurrent.GRU
elif args.rnn == 'LSTM':
  RNN = recurrent.LSTM
else:
  assert False, "Invalid RNN"

print "Creating model..."

model = Graph()
model.add_input(name="question", batch_input_shape=(args.batch_size,)+questions.shape[1:], dtype="uint")
model.add_input(name="answer", batch_input_shape=(args.batch_size,)+answers.shape[1:], dtype="uint")

if args.bidirectional:
  shared = Graph()
  shared.add_input(name="input", batch_input_shape=(args.batch_size, None), dtype="uint")
  shared.add_node(Embedding(vocab_size, args.embed_size, mask_zero=True), name="embed", input='input')
  for i in xrange(args.layers):
    shared.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True), 
        name='forward'+str(i+1), 
        input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    shared.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True, go_backwards=True), 
        name='backward'+str(i+1), 
        input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None, 
        inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
    if args.dropout > 0:
      shared.add_node(Dropout(args.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])
  shared.add_output(name='output',
      input='dropout'+str(args.layers) if args.dropout > 0 else None,
      inputs=['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else [])
else:
  shared = Sequential()
  shared.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
  for i in xrange(args.layers):
    shared.add(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True))
    if args.dropout > 0:
      shared.add(Dropout(args.dropout))

#model.add_shared_node(shared, name="shared", inputs=['question', 'answer'], merge_mode='cos', dot_axes=([1], [1]))
#model.add_node(Activation('sigmoid'), name='output', input='shared', create_output=True)
model.add_shared_node(shared, name="shared", inputs=['question', 'answer'], merge_mode='concat')
model.add_node(Dense(1, activation='sigmoid'), name='output', input='shared', create_output=True)

shared.summary()
model.summary()

print "Compiling model..."
model.compile(optimizer=args.optimizer, loss={'output':'binary_crossentropy'})

callbacks=[ModelCheckpoint(filepath=args.save_path, verbose=1, save_best_only=True), 
               EarlyStopping(patience=args.patience, verbose=1)]
model.fit({'question': questions, 'answer': answers, 'output': correct}, batch_size=args.batch_size, nb_epoch=args.epochs, 
    validation_split=args.validation_split, verbose=args.verbose, callbacks=callbacks,
    shuffle=args.shuffle if args.shuffle=='batch' else True if args.shuffle=='true' else False)
