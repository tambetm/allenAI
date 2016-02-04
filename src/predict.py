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
  texts = questions + answersA + answersB + answersC + answersD
  print "Texts size:", len(texts)

  data = text_to_data(texts, tokenizer, maxlen) 
  return data

def np_l2_normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / norm

def np_cosine_similarity(y_true, y_pred):
    assert y_true.ndim == 2
    assert y_pred.ndim == 2
    y_true = np_l2_normalize(y_true, axis=1)
    y_pred = np_l2_normalize(y_pred, axis=1)
    return np.sum(y_true * y_pred, axis=1, keepdims=False)

def convert_test_predictions(pred):
  qlen = int(pred.shape[0] / 5)
  questions = pred[0:qlen]
  answersA = pred[qlen:2*qlen]
  answersB = pred[2*qlen:3*qlen]
  answersC = pred[3*qlen:4*qlen]
  answersD = pred[4*qlen:5*qlen]
  '''
  print "Predicted vectors:", questions.shape, answersA.shape, answersB.shape, answersC.shape, answersD.shape
  print "Question:", questions[0,:4]
  print "Answer A:", answersA[0,:4]
  print "Answer B:", answersB[0,:4]
  print "Answer C:", answersC[0,:4]
  print "Answer D:", answersD[0,:4]
  '''
  sims = np.array([
    np_cosine_similarity(questions, answersA),
    np_cosine_similarity(questions, answersB),
    np_cosine_similarity(questions, answersC),
    np_cosine_similarity(questions, answersD)
  ])
  '''
  print "Similarities:", sims.shape
  print "Question 1:", sims[:,0]
  print "Question 2:", sims[:,1]
  print "Question 3:", sims[:,2]
  '''

  preds = np.argmax(sims, axis=0)
  #print "Predictions:", preds.shape
  #print "Predicted answers:", preds[:3]

  preds = [chr(ord('A') + p) for p in preds]
  #print "Predicted answers:", preds[:3]

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
  preds = convert_test_predictions(pred)

  if args.write_predictions:
    print "Writing predictions to", args.write_predictions
    write_predictions(args.write_predictions, ids, preds)

  if len(corrects) > 0:
    calculate_accuracy(preds, corrects)
