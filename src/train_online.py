import argparse
import cPickle as pickle
import numpy as np
import math
import sys
from keras.models import model_from_json
from keras.utils import generic_utils
from itertools import islice
from random import shuffle
from sklearn.metrics.pairwise import pairwise_distances
from model import *
from preprocess import *

def generate_training_data(data_path, tokenizer, model, args):
  while True:
    with open(data_path) as f:
      csv.field_size_limit(sys.maxsize)
      reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)

      while True:
        # read macrobatch_size lines from reader
        lines = list(islice(reader, args.macrobatch_size))
        #print "Lines:", len(lines)
        if not lines:
          break;

        '''
        print "Sample lines:"
        print lines[0]
        print lines[1]
        print lines[2]

        print zip(*lines[:3])
        '''
        shuffle(lines)
        ids, questions, answers = zip(*lines)
        #print "ids:", len(ids), "questions:", len(questions), "answers:", len(answers)

        texts = questions + answers
        #print "texts:", len(texts)
        data = text_to_data(texts, tokenizer, args.maxlen)
        #print "data:", data.shape

        pred = predict_data(model, data, args)
        #print "pred:", pred.shape
        half = int(pred.shape[0] / 2)
        question_vectors = pred[0:half]
        answer_vectors = pred[half:]
        #print "question_vectors:", question_vectors.shape, "answer_vectors.shape", answer_vectors.shape
        dists = pairwise_distances(question_vectors, answer_vectors, metric="cosine", n_jobs=1)
        #print "distances:", dists.shape

        X = np.empty((args.batch_size, data.shape[1]))
        y = np.empty((args.batch_size, args.hidden_size))
        n = 0
        produced = 0
        total_pa_dist = 0
        total_na_dist = 0
        total_margin = 0
        for i in xrange(len(questions)):
          sorted = np.argsort(dists[i])
          #print ""
          #print "question %d:" % i, questions[i]
          for j in sorted:
            margin = dists[i,j] - dists[i,i]
            #print "answer %d:" % j, answers[j], "(correct answer: %s)" % answers[i]
            #print "distance:", dists[i,j], "(margin %f)" % margin
            if j != i and answers[j].strip().lower() != answers[i].strip().lower() \
                and (args.min_margin is None or margin > args.min_margin):
              if (args.max_margin is None or margin < args.max_margin):
                X[n] = data[i]
                X[n+1] = data[half + i]
                X[n+2] = data[half + j]
                n += 3
                #print "Question:", questions[i], data[i]
                #print "Right answer:", answers[i], data[half + i]
                #print "Wrong answer:", answers[j], data[half + j]
                if n == args.batch_size:
                  yield X, y
                  n = 0

                total_pa_dist += dists[i,i]
                total_na_dist += dists[i,j]
                total_margin += margin
                produced += 1
              break

        if n > 0:
          yield X[:n], y[:n]

        print ""
        print "Read %d lines, used %d questions, discarded %d" % (len(lines), produced, len(lines) - produced)
        print "Average right answer distance %g, wrong answer distance %g, margin %g" % \
            (total_pa_dist / produced, total_na_dist / produced, total_margin / produced)
        print ""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("--data_path", default="/storage/hpc_tanel/allenAI/combined.txt")
  parser.add_argument("--csv_file", default="data/training_set.tsv")
  parser.add_argument("--load_tokenizer", default="model/tokenizer_studystack_full.pkl")
  parser.add_argument("--macrobatch_size", type=int, default=1000)
  parser.add_argument("--min_margin", type=float, default=0)
  parser.add_argument("--max_margin", type=float, default=0.2)
  parser.add_argument("--load_model")
  parser.add_argument("--load_arch")
  parser.add_argument("--save_arch")
  add_model_params(parser)
  add_training_params(parser)
  add_data_params(parser)
  args = parser.parse_args()
  #assert args.samples_per_epoch % args.batch_size == 0, "Samples per epoch must be divisible by batch size."

  print "Loading tokenizer..."
  tokenizer = load_tokenizer(args.load_tokenizer)
  vocab_size = vocabulary_size(tokenizer)
  print "Vocabulary size:", vocab_size

  if args.load_arch:
    print "Loading model architecture from", args.load_arch
    model = model_from_json(args.load_arch)
  else:
    print "Creating model..."
    model = create_model(vocab_size, args)

  model.summary()

  if args.save_arch:
    print "Saving model architecture to", args.save_arch
    open(args.save_arch, 'w').write(model.to_json())

  if args.load_model:
    print "Loading weights from %s" % args.load_model
    model.load_weights(args.load_model)

  print "Compiling model..."
  compile_model(model, args)

  print "Loading test data..."
  callback = TestAccuracy(args.csv_file, args)
  callback.model = model

  print "Loading training data..."
  generator = generate_training_data(args.data_path, tokenizer, model, args)

  print "Fitting model..."
  for epoch in xrange(args.epochs):
    progbar = generic_utils.Progbar(args.samples_per_epoch)
    n = 0
    while n < args.samples_per_epoch:
      X, y = next(generator)
      #print "X.shape:", X.shape
      loss = model.train_on_batch(X, y)
      bs = X.shape[0]
      progbar.add(bs, values=[('train loss', loss[0])])
      n += bs

    logs = {}
    callback.on_epoch_end(epoch, logs)

    avg_loss = progbar.sum_values['train loss'][0] / max(1, progbar.sum_values['train loss'][1])
    filename = "%s_%02d_loss_%.4f_acc_%.4f.hdf5" % (args.model_path, epoch, avg_loss, logs['val_acc'])
    print "Saving model to", filename
    model.save_weights(filename)

  print "Done"