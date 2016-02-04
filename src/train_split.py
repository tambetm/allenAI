import argparse
import cPickle as pickle
import numpy as np
from keras.models import model_from_json
from itertools import islice
from model_split import *
from preprocess import *

def load_training_data_pickle(data_path, args):
  data = pickle.load(open(data_path, "rb"))
  print "Data:", data.shape
  assert data.shape[0] % 3 == 0

  if args.nquestions is not None:
    data = data[:args.nquestions*3]

  questions = data[0::3]
  pos_answers = data[1::3]
  neg_answers = data[2::3]

  print "questions:", questions.shape, "pos_answers:", pos_answers.shape, "neg_answers:", neg_answers.shape, 
  return questions, pos_answers, neg_answers

def load_training_data_text(text_path, tokenizer, args):
  with open(text_path) as f:
    lines = list(f)
  print "Lines:", len(lines)

  if args.nquestions is not None:
    lines = lines[:args.nquestions*3]

  questions = lines[0::3]
  pos_answers = lines[1::3]
  neg_answers = lines[2::3]

  questions = text_to_data(questions, tokenizer, args.maxlen) 
  pos_answers = text_to_data(pos_answers, tokenizer, args.maxlen) 
  neg_answers = text_to_data(neg_answers, tokenizer, args.maxlen) 

  print "questions:", questions.shape, "pos_answers:", pos_answers.shape, "neg_answers:", neg_answers.shape, 
  return questions, pos_answers, neg_answers

def generate_training_data_pickle(data_path, args):
  assert args.nquestions is None, "--nquestions cannot be used with generator"

  data = load_training_data_pickle(data_path, args)
  while True:
    for i in xrange(0, data.shape[0], 3*args.batch_size):
      #print "Iteration:", i
      questions = data[i:i+3*args.batch_size:3]
      pos_answers = data[i+1:i+3*args.batch_size:3]
      neg_answers = data[i+2:i+3*args.batch_size:3]
      y = np.empty((x.shape[0], 3*args.hidden_size))
      #print "X,y:", x.shape, y.shape
      yield {'question': questions, 'pos_answer': pos_answers, 'neg_answer': neg_answers, 'output': y}

def generate_training_data_text(data_path, tokenizer, args):
  assert args.nquestions is None, "--nquestions cannot be used with generator"

  while True:
    with open(data_path) as f:
      while True:
        lines = list(islice(f, 3*args.batch_size))
        #print "Lines:", len(lines)
        if not lines:
          break;
        '''
        print "Sample lines:"
        print lines[:3]
        '''
        questions = lines[0::3]
        pos_answers = lines[1::3]
        neg_answers = lines[2::3]
        '''
        print "Sample question, pos_answer, neg_answer:"
        print questions[0]
        print pos_answers[0]
        print neg_answers[0]
        '''
        questions = text_to_data(questions, tokenizer, args.maxlen) 
        pos_answers = text_to_data(pos_answers, tokenizer, args.maxlen) 
        neg_answers = text_to_data(neg_answers, tokenizer, args.maxlen) 
        '''
        print "questions:", questions.shape, "pos_answers:", pos_answers.shape, "neg_answers:", neg_answers.shape, 
        print "Sample question, pos_answer, neg_answer:"
        print questions[0]
        print pos_answers[0]
        print neg_answers[0]
        '''
        y = np.empty((questions.shape[0], 3*args.hidden_size))
        #print "y:", y.shape
        yield {'question': questions, 'pos_answer': pos_answers, 'neg_answer': neg_answers, 'output': y}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("--data_path", default="/storage/hpc_tanel/allenAI/X_studystack_qa_cleaner_no_qm_ranking_shuffled.txt")
  parser.add_argument("--nquestions", type=int)
  parser.add_argument("--csv_file", default="data/training_set.tsv")
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
  if args.data_path.endswith(".pkl"):
    if args.generator:
      generator = generate_training_data_pickle(args.data_path, args)
    else:
      data = load_training_data_pickle(args.data_path, args)
  elif args.data_path.endswith(".txt"):
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
    history = fit_data(model, data, args)

  if args.save_history:
    print "Saving training history to", args.save_history
    pickle.dump(history, open(args.save_history, "wb"), pickle.HIGHEST_PROTOCOL)

  print "Done"