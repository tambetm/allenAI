# Archive

Historical models, that we tried during hackathon.

 * `deep1_first` - simple model with two branches for question and answer and sigmoid in the end. Trained with boolean labels, if it is correct or wrong answer to the question.
 * `deep2_triplet` - first try with cosine ranking loss, three shared branches for question, right answer and wrong answer.
 * `deep3_simple` - simplified version of the above network, questions and answers are included sequentially in batch and loss function takes it into account.
 * `deep4_rewrite` - rewrite of previous code to be more modular.
 * `deep5_negsampling` - neat idea how to perform negative sampling in parallel process using another GPU and communicate samples to training process through named FIFO. Training code didn't need to be changed at all, just pointed to FIFO file instead of original data file. But online sampling still worked better.
 * `deep6_split` - question and answers have different weights. Didn't help.
 * `deep7_lang` - after chat with Tomas Mikolov we tried simple language model to use likelihood of sentence "question answer" as measure if it makes sense. Didn't get it working, because of too big vocabulary for softmax. Didn't find existing code for hierarchical softmax.
 * `deep8_lang2` - completely flawed idea to use cosine similarity instead of cross-entropy loss in language model. The code worked, but the model didn't learn.
 * `deep9_seq2seq` - an idea to use sequence-to-sequence learning for predicting answer. Failed for the same reasons as language model.
 * `lucene1` - the first naive way of scoring answers using Lucene - just search for question+answer and see which of the options A, B, C, D results in highest max score.
 * `notebooks` - various notebooks we used for trying things out.
