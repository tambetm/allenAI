import argparse
import cPickle as pickle
import numpy as np
from keras.models import model_from_json
from itertools import islice
from model_lang2 import *
from preprocess import *

def load_training_data_text(text_path, tokenizer, args):
  with open(text_path) as f:
    lines = list(f)

  print "lines:", len(lines)
  '''
  print "Sample lines:"
  print lines[0]
  print lines[1]
  print lines[2]
  '''
  if args.nquestions is not None:
    lines = lines[:args.nquestions]

  data = text_to_data(lines, tokenizer, args.maxlen)
  inputs = data[:,:-1]  # discard the last word of answer
  outputs = data[:,1:]  # shift all words left by one

  print "inputs:", inputs.shape, "outputs:", outputs.shape
  '''
  print "Sample input, output"
  print inputs[0], outputs[0]
  print inputs[1], outputs[1]
  print inputs[2], outputs[2]
  '''
  return inputs, outputs

def generate_training_data_text(data_path, tokenizer, args):
  assert args.nquestions is None, "--nquestions cannot be used with generator"

  vocab_size = vocabulary_size(tokenizer)
  while True:
    with open(data_path) as f:
      reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
      while True:
        # read batch size lines from reader
        lines = list(islice(f, args.batch_size))
        #print "Lines:", len(lines)
        if not lines:
          break;

        data = text_to_data(lines, tokenizer, args.maxlen)
        inputs = data[:,:-1]  # discard the last word of answer
        outputs = data[:,1:]  # shift all words left by one

        #print "inputs:", inputs.shape, "outputs:", outputs.shape
        '''
        print "Sample input, output"
        print inputs[0], outputs[0]
        print inputs[1], outputs[1]
        print inputs[2], outputs[2]
        '''
        yield {'input': inputs, 'target': outputs, 'output': np.empty((inputs.shape[0], inputs.shape[1], 2*args.embed_size))}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("--data_path", default="/storage/hpc_tanel/allenAI/studystack_qa_cleaner_no_qm.txt")
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