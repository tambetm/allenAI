import argparse
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument("save_path")
parser.add_argument("--data", default="/storage/allenAI/X_studystack_qa_cleaner_ranking.txt")
parser.add_argument("--max_words", type=int)
parser.add_argument("--question_maxlen", type=int)
parser.add_argument("--answer_maxlen", type=int)
args = parser.parse_args()

print "Loading data..."
questions = []
corrects = []
incorrects = []
i = 0
with open(args.data) as f:
  for line in f:
    if i % 3 == 0:
      questions.append(line)
    elif i % 3 == 1:
      corrects.append(line)
    elif i % 3 == 2:
      incorrects.append(line)
    else:
      assert False
    i += 1

print "Questions:", len(questions), "Corrects:", len(corrects), "Incorrects:", len(incorrects)

print "Sample questions, corrects, incorrects:"
for i in xrange(3):
    print questions[i], corrects[i], incorrects[i]

print "Tokenizing data..."
tokenizer = Tokenizer(args.max_words)
tokenizer.fit_on_texts(questions)
print "Number of words in questions: ", len(tokenizer.word_index)
tokenizer.fit_on_texts(corrects)
print "Number of words with correct answers: ", len(tokenizer.word_index)
tokenizer.fit_on_texts(corrects)
print "Number of words with incorrect answers: ", len(tokenizer.word_index)

wcounts = tokenizer.word_counts.items()
wcounts.sort(key=lambda x: x[1], reverse=True)
print "Most frequent words:", wcounts[:10]
print "Most rare words:", wcounts[-10:]

print "Number of words occurring %d times:" % wcounts[-1][1], np.sum(np.array(tokenizer.word_counts.values())==wcounts[-1][1])

print "Converting text to sequences..."
questions_seq = tokenizer.texts_to_sequences(questions)
corrects_seq = tokenizer.texts_to_sequences(corrects)
incorrects_seq = tokenizer.texts_to_sequences(incorrects)

print "Sample sequences:"
for i in xrange(3):
  print questions_seq[i], corrects_seq[i], incorrects_seq[i]

pickle.dump([questions_seq, corrects_seq, incorrects_seq], open("sequences.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

if args.question_maxlen:
  question_maxlen = args.question_maxlen
else:
  question_maxlen = max([len(q) for q in questions_seq])
if args.answer_maxlen:
  answer_maxlen = args.answer_maxlen
else:
  answer_maxlen = max([len(a) for a in corrects_seq]+[len(a) for a in incorrects_seq])
print "Questions maxlen:", question_maxlen, "Answers maxlen:", answer_maxlen

print "Padding sequences..."
questions_seq_pad = pad_sequences(questions_seq, maxlen=question_maxlen) 
corrects_seq_pad = pad_sequences(corrects_seq, maxlen=answer_maxlen)
incorrects_seq_pad = pad_sequences(incorrects_seq, maxlen=answer_maxlen)

print "Sample padded sequences:"
print questions_seq_pad[0]
print corrects_seq_pad[0]
print incorrects_seq_pad[0]

print "Saving results..."
pickle.dump([questions_seq_pad, corrects_seq_pad, incorrects_seq_pad], open(args.save_path, "wb"), pickle.HIGHEST_PROTOCOL)

print "Done"
