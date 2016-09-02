#! /usr/bin/env python

import os
import time
import datetime
import cPickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn import TextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("rb_lambda", 1e-2, "Robust Regularizaion lambda (default: 1e-2)")
tf.flags.DEFINE_float("alpha", 0.1, "Data noise level alpha(default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200 --> 100 by lrank)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
print("Vocabulary Size: {:d}".format(len(vocabulary)))
# Randomly shuffle data
np.random.seed(1001003)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

print "loading word2vec vectors..."
Dict = data_helpers.load_bin_vec("./data/GoogleNews-vectors-negative300.bin", vocabulary_inv)
print "word2vec loaded!"
print "num words already in word2vec: " + str(len(Dict))
data_helpers.add_unknown_words(Dict, vocabulary_inv)
cPickle.dump([Dict], open("mr.p", "wb") )

tmp = cPickle.load( open("mr.p", "rb") )
Dict = tmp[0]
w2v = []
for x in range(0, len(Dict), 1):
    w2v.append( Dict[vocabulary_inv[x]].tolist() )


length = len(x_shuffled)
score_sum = []
best_score = 0

for cv in range(0, 10):
    print "CV:", cv

    x_train = []
    x_dev = []
    y_train = []
    y_dev = []
    for i in range(0, length):
        index = i % 10
        if index == cv:
            x_dev.append( x_shuffled[i] )
            y_dev.append( y_shuffled[i] )
        else:
            x_train.append( x_shuffled[i] )
            y_train.append( y_shuffled[i] )

    x_train = np.array( x_train )
    y_train = np.array( y_train )
    x_dev = np.array( x_dev )
    y_dev = np.array( y_dev )

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    best_score = 0

    #noisy the dev data
    for item in x_dev:
        for l in xrange(len(item)):
            if np.random.random_sample() < FLAGS.alpha:
                item[l] = 0

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement,
          intra_op_parallelism_threads=2,
          inter_op_parallelism_threads=2)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                #dict=w2v,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                rb_lambda=FLAGS.rb_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            sess.run(cnn.W.assign(w2v))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                  cnn.is_add_noise: True,
                  cnn.pad_num: vocabulary["<PAD/>"]
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0,
                  cnn.is_add_noise: False,
                  cnn.pad_num: vocabulary["<PAD/>"]
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                global best_score
                if accuracy > best_score:
                    best_score = accuracy
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    score_sum.append( best_score )


sco = 0
for s in score_sum:
    sco += s
print score_sum
print "CV score:", sco / 10
