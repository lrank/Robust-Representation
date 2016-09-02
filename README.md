# Robust-Representation


##Intro
This code is an implementation the following work:

Learning Robust Representation of Text. Li, Yitong, Trevor Cohn and Timothy Baldwin. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016).

The code implements high-order derivative as regularization to learning robust model aganist noise.

The CNN-mode is following [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) and [Denny Britz](http://www.wildml.com)'s implementation(https://github.com/dennybritz/cnn-text-classification-tf) in tensorflow().

We evaluate on Pang and Lee's movie review dataset (MR).
More data in the paper can be found at [HarvardNLP](https://github.com/harvardnlp/sent-conv-torch/).
Please refer the original paper when using the data.

And you might need a pre-train word embeddings(https://code.google.com/archive/p/word2vec/).

##Requrements

- Python 2.7 or 3
- Tensorflow
- Numpy

##Running the model

>>python train.py
