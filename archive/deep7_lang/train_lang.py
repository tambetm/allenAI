import argparse
import cPickle as pickle
import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from itertools import islice
from model_lang import *
from preprocess import *

def seq_to_categorical(y, nb_classes):
  yy = y.reshape((-1))
  yy = to_categorical(yy, nb_classes)
  yy = yy.reshape((-1, y.shape[1], yy.shape[1]))
  return yy

def load_training_data_text(text_path, tokenizer, args):
  texts = []
  targets = []
  with open(text_path) as f:
    reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
    for line in reader:
      question = line[1]
      answer = line[2]
      texts.append(question + " " + answer)
      targets.append(answer)

  print "Texts:", len(texts), "Targets:", len(targets)
  '''
  print "Sample texts and targets:"
  print texts[0], targets[0]
  print texts[1], targets[1]
  print texts[2], targets[2]
  '''

  if args.nquestions is not None:
    texts = texts[:args.nquestions]
    targets = targets[:args.nquestions]

  text_sequences = tokenizer.texts_to_sequences(texts)
  inputs = pad_sequences(text_sequences, maxlen=args.maxlen + 1)
  outputs = inputs[:,1:]  # shift all words left by one
  inputs = inputs[:,:-1]  # discard the last word of answer

  #target_sequences = tokenizer.texts_to_sequences(targets)
  #outputs = pad_sequences(target_sequences, maxlen=args.maxlen)
  outputs = seq_to_categorical(outputs, vocabulary_size(tokenizer))
  
  '''
  masks = np.zeros(inputs.shape, dtype="uint")
  for i, t in enumerate(target_sequences):
    masks[i, args.maxlen - len(t):] = 1
  '''
  print "inputs:", inputs.shape, "outputs:", outputs.shape #, "masks:", masks.shape
  '''
  print "Sample input, output, mask"
  print inputs[0], outputs[0], masks[0]
  print inputs[1], outputs[1], masks[1]
  print inputs[2], outputs[2], masks[2]
  '''
  return inputs, outputs #, masks

def generate_training_data_text(data_path, tokenizer, args):
  assert args.nquestions is None, "--nquestions cannot be used with generator"

  vocab_size = vocabulary_size(tokenizer)
  while True:
    with open(data_path) as f:
      reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
      while True:
        # read batch size lines from reader
        lines = list(islice(reader, args.batch_size))
        #print "Lines:", len(lines)
        if not lines:
          break;

        # read texts and targets
        texts = []
        targets = []
        for line in lines:
          question = line[1]
          answer = line[2]
          texts.append(question + " " + answer)
          targets.append(answer)
        #print "Texts:", len(texts), "Targets:", len(targets)
        '''
        print "Sample texts and targets:"
        print texts[0], targets[0]
        print texts[1], targets[1]
        print texts[2], targets[2]
        '''

        #print "converting texts to sequences to matrix"
        text_sequences = tokenizer.texts_to_sequences(texts)
        inputs = pad_sequences(text_sequences, maxlen=args.maxlen + 1)
        outputs = inputs[:,1:]  # shift all words left by one
        inputs = inputs[:,:-1]  # discard the last word of answer

        #print "converting targets to sequences to matrix"
        #target_sequences = tokenizer.texts_to_sequences(targets)
        #outputs = pad_sequences(target_sequences, maxlen=args.maxlen)
        outputs = seq_to_categorical(outputs, vocabulary_size(tokenizer))
        
        #print "creating masks masks for targets"
        '''
        masks = np.zeros(inputs.shape, dtype="uint")
        for i, t in enumerate(target_sequences):
          masks[i, args.maxlen - len(t):] = 1
        '''
        print "inputs:", inputs.shape, "outputs:", outputs.shape #, "masks:", masks.shape

        '''
        print "Sample input, output, mask"
        print inputs[0], outputs[0], masks[0]
        print inputs[1], outputs[1], masks[1]
        print inputs[2], outputs[2], masks[2]
        '''

        yield {'input': inputs, 'output': outputs} #, 'mask': masks}

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