# Robust-Representation


##Introduction
This code is an implementation the following work:

Learning Robust Representation of Text. Li, Yitong, Trevor Cohn and Timothy Baldwin. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016).

The code implements high-order derivative as regularization to learning robust model aganist noise based on Convolutional Neural Network (CNN) model.

###CNN implementations
The CNN-model is following YoonKim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) and Denny Britz's implementation (https://github.com/dennybritz/cnn-text-classification-tf) in tensorflow.

###Data
We evaluate on Pang and Lee's movie review dataset (MR) using 10-fold cross validation.

More data in the paper can be found at [HarvardNLP](https://github.com/harvardnlp/sent-conv-torch/tree/master/data).

And you can download a pre-trained word embeddings [word2vec](https://code.google.com/archive/p/word2vec/).

Please refer the original paper for details.

##Requrements

- Python 2.7 or 3
- Tensorflow
- Numpy

##Running the model

Running the model using following commnad:
>> python train.py []

For exmaple, training using robust regularization only:
>> python train.py --rb_lambda 1e-2  --dropout_keep_prob 1

training models with dropout only:
>> python train.py --dropout_keep_prob 0.5 --rb_lambda 0
