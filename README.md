# allenAI

This is the code that got our team Cappucino Monkeys 4th place at [The Allen AI Science Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge/leaderboard) Kaggle competition.

In this competition your program has to do well in standardized 8th grade science exam. The exam consists of multiple choice questions spanning biology, chemistry, physics, math and so on. Each question has four possible answers, of which exactly one is correct.

We ended up ensembling 5 deep learning models and 3 information retrieval models. You will find the code to replicate most of the results below.

## Dataset

Firstly the crucial part of our solution was the dataset. We collected 5M question-answer pairs from websites such as [Studystack](http://www.studystack.com/). You can download part of the dataset below:

 * [Studystack](https://drive.google.com/file/d/0B0fFJSGDUPcgUFJpTVl3QXhnNTQ/view?usp=sharing) (454743 questions)

The same data was used both for training deep learning models and in information retrieval approach.

## Deep Learning Model

Our deep learning approach was inspired by [IBM Watson LSTM paper]. We used RNN (GRU) to get sentence vector for both question and answer. Then we used cosine ranking loss to make cosine similarity for the question and the right answer bigger by some margin than for the wrong answer. One nice trick was, that we didn't have 3 separate (shared) branches in the network, but instead used one network and combined questions, right answers and wrong answers sequentially in batch. Loss function considered every third sample in batch as question, every other third as right answer and another third as wrong answer. This simplified the network architecture greatly and made it possible to use it in prediction phase with any number of answers.

As our dataset had only right answers, we needed to produce wrong answers ourselves. This is called "negative sampling" (right answer is the "positive sample" and wrong answer is the "negative sample"). We used strategy similar to [Google FaceNet](http://arxiv.org/abs/1503.03832) to choose wrong answers from the same (macro)batch. For the network to learn well, it is useful to not feed random wrong answers to it, but those that are "hard". That means answers that are wrong, but close to question in the cosine distance. But if you feed only the hardest questions to the network, it fails to converge, it is just "too hard". Instead people have found that using "semi-hard negative samples" - answers, which are further than right answer, but still within the margin - works best. And that's what we did.

The architecture of the final network:
 * embedding layer: size 100
 * recurrent layer: GRU, size 512
 * no dropout
 * margin: 0.2
 * batch size: 300 (100 questions)
 * optimizer: Adam
 * samples per epoch: 1500000 (500000 questions)
 * max epochs: 100
 * patience: 10 (stop training if validation accuracy does not improve in this number of epochs)

Another important feature was that after each epoch we tested our model on Allen AI training set and included the accuracy in the file name of saved weights. This allowed to track much easier how the currently trained models are doing and which can be killed to make room for subsequent experiments. At some point we had 10 GPUs running different experiments.

We used excellent [Keras toolkit](http://keras.io/) for implementing the model.

### How to run the training

To run the training with deep model:
```
python deep/train_online.py model/test
```

The only required parameter is the path where to save the models. In this example the file names will be something like `model/test_00_loss_0.1960_acc_0.2868.hdf5`, where `00` is the epoch number (starting from 0), `0.1960` is the training loss and `0.2868` is the accuracy on validation set (usually Allen AI training set).

Other parameters:
 * `--data_path` - path to data file. It should be text file with id, question and answer on each line, separated by tab.
 * `--csv_file` - path to validation set, usually Allen AI training set file, which is default.
 * `--load_tokenizer` - the tokenizer to use, determines the number of words in vocabulary and embedding layer size. Default is Studystack full vocabulary tokenizer.
 * `--maxlen` - maximum length of sentences (in words), default is 255.
 * `--macrobatch_size` - size of macrobatch for negative sampling. Default is 1000.
 * `--min_margin` - minimal margin for negative sampling, default is 0 (semi-hard).
 * `--max_margin` - maximal margin for negative sampling, default is 0.2 (when margin is bigger than that, then gradient is 0 and the sample is basically useless for training).
 * `--load_model` - start training from pre-trained model.
 * `--rnn` - either `GRU` or `LSTM`. Default is GRU, which seemed to converge faster in our experiments.
 * `--embed_size` - size of embedding layer, default is 100.
 * `--hidden_size` - size of hidden (RNN) layer, default is 512.
 * `--layers` - number of hidden layers, default is 1.
 * `--dropout` - dropout fraction (how much to drop), applied after every hidden layer.
 * `--bidirectional` -- use bidirectional RNN, works both with GRU and LSTM.
 * `--batch_size` -- minibatch size, default is 300 (100 questions), must be multiple of 3.
 * `--margin` -- margin to use in cosine ranking loss, default is 0.2.
 * `--optimizer` - either `adam`, `rmsprop` or `sgd`. Adam is the default.
 * `--lr` - learning rate to use. Works only when combined with `--lr_epochs`.
 * `--lr_epochs` - halve learning rate after every this number of epochs.
 * `--samples_per_epoch` - number of samples per epoch, default is 1500000 (500000 questions), should be multiple of 3.
 * `--epochs` - maximum number of epochs.
 * `--patience` - stop training when validation set accuracy hasn't improved for this number of epochs.

### How to run the prediction

To run prediction with deep model:
```
python src/predict.py model/test_00_loss_0.1960_acc_0.2868.hdf5
```

By default it just calculates accuracy on Allen AI training set. You may want to use following options:
 * `--csv_file` - the file to calculate predictions for, default is Allen AI training set, but can also be `data/validation_set.tsv` or `data/test_set.tsv`. The code handles the missing "correct" column automatically.
 * `--write_predictions` - write predictions in CSV file. The first column is question id, the second is predicted answer,  followed by scores (cosine similarities) for all answers A, B, C and D. The file format was supposed to be compatible with Kaggle submission, but for ensembling purposes we had to include scores too.

Many options for the training script apply as well. For example you need to match the hidden layer size and number of layers with saved model, otherwise it will give an error.

### How to run the preprocessing

The preprocessing is mostly used to produce tokenizer. Tokenizer determines the vocabulary size and embedding layer size in network. We used Keras default tokenizer class, which proved sufficient for our purposes. There is actually no need to run preprocessing, because the tokenizer for default dataset is already included in the repository. The instructions are included here just in case you want to use your own dataset.

First you need to get rid of the id field in original data file, so it doesn't clutter the vocabulary:
```
cut -f2,3 data/studystack_qa_cleaner_no_qm.txt >data/studystack.txt
```

Then run preprocessing on it:
```
python src/preprocess.py data/studystack --save_tokenizer model/tokenizer_studystack.pkl
```

Additional options:
 * `--max_words` - limit the vocabulary to this number of most frequent words. Beware that if you limit the number of words, Keras default tokenizer just removes all the rare words from sentences, instead of replacing them with "UNKNOWN".
