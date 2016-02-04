import argparse
import csv
from model import *
from preprocess import *

def load_test_data(csv_file):
  with open(csv_file) as f:
    reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
    header = next(reader)  # ignore header
    is_train_set = (len(header) == 7)
    if is_train_set:
      ids, questions, corrects, answersA, answersB, answersC, answersD = zip(*list(reader))
    else:
      ids, questions, answersA, answersB, answersC, answersD = zip(*list(reader))
      corrects = []
  print "Questions: ", len(questions)
  assert len(questions) == len(answersA) == len(answersB) == len(answersC) == len(answersD)
  assert not is_train_set or len(corrects) == len(questions)

  print "Sample question and answers:"
  for i in xrange(3):
    print questions[i], "A:", answersA[i], "B:", answersB[i], "C:", answersC[i], "D:", answersD[i], "Correct: ", corrects[i] if is_train_set else '?'

  return ids, questions, answersA, answersB, answersC, answersD, corrects

def convert_test_data(questions, answersA, answersB, answersC, answersD, tokenizer, maxlen):
  texts = []
  for i, q in enumerate(questions):
    texts.append(q + " " + answersA[i])
    texts.append(q + " " + answersB[i])
    texts.append(q + " " + answersC[i])
    texts.append(q + " " + answersD[i])
  print "Texts size:", len(texts)

  data = text_to_data(texts, tokenizer, maxlen)
  print "Data:", data.shape
  inputs = data[:,:-1]  # discard the last word of answer
  outputs = data[:,1:]  # shift all words left by one
  print "Inputs:", inputs.shape, "Outputs:", outputs.shape
  return inputs, outputs

def np_l2_normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / norm

def np_cosine_similarity(y_true, y_pred):
    assert y_true.ndim == 3
    assert y_pred.ndim == 3
    y_true = np_l2_normalize(y_true, axis=2)
    y_pred = np_l2_normalize(y_pred, axis=2)
    return np.mean(np.sum(y_true * y_pred, axis=2), axis=1, keepdims=False)

def convert_test_predictions(pred, args):
  print "pred:", pred.shape
  y_pred = pred[:,:,:args.embed_size]
  y_true = pred[:,:,args.embed_size:]
  print "y_pred:", y_pred.shape, "y_true:", y_true.shape

  print "y_pred=", y_pred[0,0]
  print "y_true=", y_true[0,0]

  sims = np_cosine_similarity(y_true, y_pred)
  print "sims:", sims.shape
  sims = sims.reshape((-1, 4))
  print "after reshape:", sims.shape
  print "Similarities:"
  print sims[0]
  print sims[1]
  print sims[2]

  preds = np.argmax(sims, axis=1)
  print "Predictions:", preds.shape
  print "Predicted answers:", preds[:3]

  preds = [chr(ord('A') + p) for p in preds]
  print "Predicted answers:", preds[:3]

  return preds

def calculate_accuracy(preds, corrects):
  correct = sum([corrects[i] == p for i,p in enumerate(preds)])
  accuracy = float(correct) / len(preds)
  print "Correct: %d Total: %d Accuracy: %f" % (correct, len(preds), accuracy)
  return accuracy

def write_predictions(file_path, ids, preds):
  with open(file_path, "w") as f:
    f.write("id,correctAnswer\n")
    for i in xrange(len(preds)):
      f.write("%s,%s\n" % (ids[i], preds[i]))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("--csv_file", default="data/training_set.tsv")
  parser.add_argument("--load_tokenizer", default="model/tokenizer_studystack_full.pkl")
  parser.add_argument("--vocab_size", type=int)
  parser.add_argument("--load_arch")
  parser.add_argument("--write_predictions")
  add_model_params(parser)
  add_data_params(parser)
  add_training_params(parser)
  args = parser.parse_args()

  print "Loading tokenizer..."
  tokenizer = load_tokenizer(args.load_tokenizer)
  vocab_size = vocabulary_size(tokenizer)
  if args.vocab_size:
    print "Overriding original vocabulary size", vocab_size
    vocab_size = args.vocab_size
  print "Vocabulary size:", vocab_size

  print "Loading data..."
  ids,q,a,b,c,d,corrects = load_test_data(args.csv_file)
  data = convert_test_data(q,a,b,c,d,tokenizer,args.maxlen)

  if args.load_arch:
    print "Loading model architecture from", args.load_arch
    model = model_from_json(args.load_arch)
  else:
    print "Creating model..."
    model = create_model(vocab_size, args)

  model.summary()

  print "Loading weights from %s" % args.model_path
  model.load_weights(args.model_path)

  print "Compiling model..."
  compile_model(model, args)

  print "Predicting..."
  pred = predict_data(model, data, args)
  preds = convert_test_predictions(pred, args)

  if args.write_predictions:
    print "Writing predictions to", args.write_predictions
    write_predictions(args.write_predictions, ids, preds)

  if len(corrects) > 0:
    calculate_accuracy(preds, corrects)
