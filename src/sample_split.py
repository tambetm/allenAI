import argparse
import csv
from random import shuffle
from model_split import *
from preprocess import *
from sklearn.metrics.pairwise import pairwise_distances

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("model_path")
  parser.add_argument("output_path")
  parser.add_argument("--data_path", default="/storage/hpc_tanel/allenAI/studystack_qa_cleaner_no_qm.txt")
  parser.add_argument("--load_tokenizer", default="model/tokenizer_studystack_full.pkl")
  parser.add_argument("--macrobatch_size", type=int, default=1000)
  parser.add_argument("--min_margin", type=float)
  parser.add_argument("--max_margin", type=float)
  parser.add_argument("--load_arch")
  parser.add_argument("--save_arch")
  add_model_params(parser)
  add_training_params(parser)
  add_data_params(parser)
  args = parser.parse_args()

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

  print "Loading weights from %s" % args.model_path
  model.load_weights(args.model_path)

  print "Compiling model..."
  compile_model(model, args)

  print "Sampling data to", args.output_path
  output = open(args.output_path, "a")
  while True:
    with open(args.data_path) as f:
      reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)

      while True:
        # read macrobatch_size lines from reader
        lines = list(islice(reader, args.macrobatch_size))
        print "Lines:", len(lines)
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
        print "ids:", len(ids), "questions:", len(questions), "answers:", len(answers)

        data_questions = text_to_data(questions, tokenizer, args.maxlen)
        data_answers = text_to_data(answers, tokenizer, args.maxlen)
        data = [data_questions, data_answers, data_answers]
        print "data_questions:", data_questions.shape, "data_answers:", data_answers.shape

        pred = predict_data2(model, data, args)
        print "pred:", pred.shape
        half = int(pred.shape[1] / 3)
        question_vectors = pred[:,0:half]
        answer_vectors = pred[:,half:2*half]
        print "question_vectors:", question_vectors.shape, "answer_vectors.shape", answer_vectors.shape
        dists = pairwise_distances(question_vectors, answer_vectors, metric="cosine", n_jobs=1)
        print "distances:", dists.shape

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
                output.write(questions[i]+"\n")
                output.write(answers[i]+"\n")
                output.write(answers[j]+"\n")
              else:
                print "discarded"
              break

        output.flush()
