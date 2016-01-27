import argparse
import cPickle as pickle
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge
from keras.layers import recurrent
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("--data", default="data.pkl")
parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="LSTM")
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--validation_split", type=float, default=0.05)
parser.add_argument("--shuffle", choices=['batch', 'true', 'false'], default='true')
parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='rmsprop')
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()

print "Loading data..."
questions, answers, correct = pickle.load(open(args.data, "rb"))
assert questions.shape[0] == answers.shape[0] == correct.shape[0]

vocab_size = max(np.max(questions), np.max(answers)) + 1
print "Vocabulary size:", vocab_size, "Questions: ", questions.shape, "Answers: ", answers.shape, "Correct: ", correct.shape

if args.rnn == 'GRU':
  RNN = recurrent.GRU
elif args.rnn == 'LSTM':
  RNN = recurrent.LSTM
else:
  assert False, "Invalid RNN"

print "Creating model..."

qrnn = Sequential()
qrnn.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
qrnn.add(RNN(args.hidden_size, return_sequences=False))

arnn = Sequential()
arnn.add(Embedding(vocab_size, args.embed_size, mask_zero=True))
arnn.add(RNN(args.hidden_size, return_sequences=False))

model = Sequential()
model.add(Merge([qrnn, arnn], mode='concat'))
model.add(Dense(1, activation='sigmoid'))

#model.summary()

print "Compiling model..."
model.compile(optimizer=args.optimizer, loss='binary_crossentropy', class_mode='binary')

callbacks=[ModelCheckpoint(filepath=args.save_path, verbose=1, save_best_only=True), 
               EarlyStopping(patience=args.patience, verbose=1)]
model.fit([questions, answers], correct, batch_size=args.batch_size, nb_epoch=args.epochs, 
    validation_split=args.validation_split, show_accuracy=True,
    shuffle=args.shuffle if args.shuffle=='batch' else True if args.shuffle=='true' else False, 
    verbose=args.verbose, callbacks=callbacks)
