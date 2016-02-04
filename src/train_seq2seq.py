import argparse
import cPickle as pickle
import csv
import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from itertools import islice
from model_seq2seq import *
from preprocess import *

def load_training_data_text(text_path, tokenizer, args):
  with open(text_path) as f:
    reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
    lines = list(reader)
    print "Lines:", len(lines)

    if args.nquestions is not None:
      lines = lines[:args.nquestions]

    data = zip(*lines)

  questions = text_to_data(data[1], tokenizer, args.maxlen) 
  answers = text_to_data(data[2], tokenizer, args.maxlen + 1) 

  print "questions:", questions.shape, "answers:", answers.shape
  return questions, answers

def generate_training_data_text(data_path, tokenizer, args):
  assert args.nquestions is None, "--nquestions cannot be used with generator"

  vocab_size = vocabulary_size(tokenizer)
  while True:
    with open(data_path) as f:
      reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
      while True:
        lines = list(islice(reader, args.batch_size))
        print "Lines:", len(lines)
        if not lines:
          break;
        '''
        print "Sample lines:"
        print lines[:3]
        '''
        data = zip(*lines)
        questions = data[1]
        answers = data[2]

        print "Sample question, answer:"
        print questions[0], answers[0]
        print questions[1], answers[1]
        print questions[2], answers[2]

        questions = tokenizer.texts_to_sequences(questions)
        answers = tokenizer.texts_to_sequences(answers)
        questions = pad_sequences(questions, maxlen=args.maxlen, padding='pre', truncating='pre') 
        answers = pad_sequences(answers, args.maxlen + 1, padding='post', truncating='post') 

        print "questions:", questions.shape, "answers:", answers.shape 
        print "Sample question, answer:"
        print questions[0], answers[0]
        print questions[1], answers[1]
        print questions[2], answers[2]

        targets1 = answers[:,1:]
        answers = answers[:,:-1]
        print "targets1:", targets1.shape
        targets2 = targets1.reshape((-1))
        print "targets2:", targets2.shape
        targets3 = to_categorical(targets2, vocab_size)
        print "targets3:", targets3.shape
        targets4 = targets3.reshape((-1, args.maxlen, vocab_size))
        print "targets4:", targets4.shape

        print "Sample targets:"
        print targets1[0]
        print targets4[0]
        print np.argmax(targets4[0], axis=1)

        print targets1[1]
        print targets4[1]
        print np.argmax(targets4[1], axis=1)

        yield {'question': questions, 'answer': answers, 'output': targets4}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("--data_path", default="/storage/hpc_tanel/allenAI/studystack_qa_cleaner_no_qm.txt")
  parser.add_argument("--nquestions", type=int)
  parser.add_argument("--csv_file") #, default="data/training_set.tsv")
  parser.add_argument("--load_tokenizer", default="model/tokenizer_studystack_full.pkl")
  parser.add_argument("--load_arch")
  parser.add_argument("--save_arch")
  parser.add_argument("--generator", action="store_true", default=False)
  parser.add_argument("--save_history")
  add_model_params(parser)
  add_training_params(parser)
  add_data_params(parser)
  args = parser.parse_args()
  #assert args.samples_per_epoch % args.batch_size == 0, "Samples per epoch must be divisible by batch size."

  print "Loading tokenizer..."
  tokenizer = load_tokenizer(args.load_tokenizer)
  vocab_size = vocabulary_size(tokenizer)
  print "Vocabulary size:", vocab_size

  print "Loading data..."
  if args.generator:
    generator = generate_training_data_text(args.data_path, tokenizer, args)
  else:
    data = load_training_data_text(args.data_path, tokenizer, args)

  if args.load_arch:
    print "Loading model architecture from", args.load_arch
    model = model_from_json(args.load_arch)
  else:
    print "Creating model..."
    model = create_model(vocab_size, args)

  next(generator)
  assert False

  model.summary()

  if args.save_arch:
    print "Saving model architecture to", args.save_arch
    open(args.save_arch, 'w').write(model.to_json())

  print "Compiling model..."
  compile_model(model, args)

  print "Fitting model..."
  if args.generator:
    history = fit_generator(model, generator, args)
  else:
    history = fit_data(model, data, vocab_size, args)

  if args.save_history:
    print "Saving training history to", args.save_history
    pickle.dump(history, open(args.save_history, "wb"), pickle.HIGHEST_PROTOCOL)

  print "Done"