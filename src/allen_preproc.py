import argparse
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("--data", default="/storage/hpc_tanel/allenAI/X_studystack_qa_cleaner.txt")
parser.add_argument("--max_words", type=int)
parser.add_argument("--question_maxlen", type=int)
parser.add_argument("--answer_maxlen", type=int)
args = parser.parse_args()

print "Loading data..."
questions = []
answers = []
corrects = []
with open(args.data) as f:
  reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)
  for correct, question, answer in reader:
    corrects.append(correct)
    questions.append(question)
    answers.append(answer)

print "Questions:", len(questions), "Answers:", len(answers), "Corrects:", len(corrects)

print "Sample questions, answers, correct/notcorrect:"
for i in xrange(3):
    print questions[i], answers[i], corrects[i]

print "Tokenizing data..."
tokenizer = Tokenizer(args.max_words)
tokenizer.fit_on_texts(questions)
print "Number of words in questions: ", len(tokenizer.word_index)
tokenizer.fit_on_texts(answers)
print "Number of words with answers: ", len(tokenizer.word_index)

wcounts = tokenizer.word_counts.items()
wcounts.sort(key=lambda x: x[1], reverse=True)
print "Most frequent words:", wcounts[:10]
print "Most rare words:", wcounts[-10:]

print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

print "Converting text to sequences..."
questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)

print "Sample sequences:"
for i in xrange(3):
  print questions_seq[i], answers_seq[i]

pickle.dump([questions_seq, answers_seq], open("sequences.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

if args.question_maxlen:
  question_maxlen = args.question_maxlen
else:
  question_maxlen = max([len(q) for q in questions_seq])
if args.answer_maxlen:
  answer_maxlen = args.answer_maxlen
else:
  answer_maxlen = max([len(a) for a in answers_seq])
print "Questions maxlen:", question_maxlen, "Answers maxlen:", answer_maxlen

print "Padding sequences..."
questions_seq_pad = pad_sequences(questions_seq, maxlen=question_maxlen) 
answers_seq_pad = pad_sequences(answers_seq, maxlen=answer_maxlen)

print "Sample padded sequences:"
print questions_seq_pad[0]
print answers_seq_pad[0]

print "Saving results..."
pickle.dump([questions_seq_pad, answers_seq_pad, np.array(corrects)], open(args.save_path, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"
