#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 10:01:27 2018

@author: shawn
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import BatchDatasetReader as BDR
import read_Data_list as RDL
import sys
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# path variable
logs_dir = 'logs/'
data_dir = '/home/lxw/tensorflowproject/data/idrid'

# basic constant variable
IMG_SIZE = 640
num_of_classes = 2
print_freq = 10
WIDTH = 4288
HEIGHT = 2848
# training constant variable
MAX_EPOCH = 500
batch_size = 1
test_batchsize = 1
train_nbr = 54
test_nbr = 27
gama = 64
step_every_epoch = int(train_nbr / batch_size)
test_every_epoch = int(test_nbr / test_batchsize)
learningrate = 1.0e-4 #tf.Variable(1e-4, dtype=tf.float32)
learningrateend = 1.0e-6
# the parameters of aupr
range_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# range_threshold = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# flags parameters
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


# data_dir = "Data/"
class Unet:
    def __init__(self, img_rows=IMG_SIZE, img_cols=IMG_SIZE):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data_util(self):
        image_options = {'resize': True, 'resize_size': IMG_SIZE}  # resize all your images
        train_records, valid_records = RDL.read_dataset(data_dir)  # get read lists
        train_dataset_reader = BDR.BatchDatset(train_records, image_options,augmentation=True)
        validation_dataset_reader = BDR.BatchDatset(valid_records, image_options,augmentation=False)
        return train_dataset_reader, validation_dataset_reader

    def model(self, image, is_train=True, reuse=False):
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            #W_init = tf.contrib.layers.xavier_initializer()
            W_init = tf.contrib.layers.variance_scaling_initializer()
            # W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
            net = tl.layers.InputLayer(image, name='input_layer')  # input image

            # filter = [96 for _ in range(10)]
            filter = [16, 64, 64, 128, 128, 128, 256]
            concat={}
            i = 0
            block = 1
            depth = 6
            for j in range(1, depth):
                i = i + 1
                net = tl.layers.Conv2d(net, filter[j], (3, 3), (1, 1),
                                       act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_' + str(i))
                i = i+1
                net = tl.layers.Conv2d(net, filter[j], (3, 3), (1, 1),
                                       act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_' + str(i))
                concat[j] = net
                net = tl.layers.AtrousConv2dLayer(net, filter[j], (3, 3), 2,
                                                  act=tf.nn.relu, padding='SAME', W_init=W_init, name='atro_conv'+str(block))
                block = block + 1

            i = i + 1
            net = tl.layers.Conv2d(net, filter[block], (3, 3), (1, 1),
                                   act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_' + str(i))
            i = i + 1
            net = tl.layers.Conv2d(net, filter[block], (3, 3), (1, 1),
                                   act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_' + str(i))

            for j in range(depth-1, 0, -1):
                i = i + 1
                net = tl.layers.Conv2d(net, filter[j], (3, 3), (1, 1),
                                       act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_' + str(i))
                net = tl.layers.ConcatLayer([net, concat[j]], 3, name='concat_'+str(j))
                i = i+1
                net = tl.layers.Conv2d(net, filter[j], (3, 3), (1, 1),
                                       act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_'+str(i))
                i = i+1
                net = tl.layers.Conv2d(net, filter[j], (3, 3), (1, 1),
                                       act=tf.nn.relu, padding='SAME', W_init=W_init, name='conv_'+str(i))
            i = i+1
            net = tl.layers.Conv2d(net, num_of_classes, (1, 1), (1, 1),
                                   padding='SAME', W_init=W_init, name='conv_'+str(i))
            y = net.outputs  # transfer tl object to logits tensor
            pred = tf.argmax(y, 3, name="prediction")

        return pred, y, net

    def loss(self, logits, annotation):
        positive_count = tf.count_nonzero(annotation,dtype=tf.float32)
        total_num = tf.constant(IMG_SIZE * IMG_SIZE,dtype=tf.float32)
        negativa_count = tf.subtract(total_num, positive_count)
        alpha = tf.div(tf.multiply(negativa_count,1.0),(positive_count*gama))

        classes_weights = [1.0, alpha]
        labels = tf.squeeze(annotation, squeeze_dims=[3])
        # labels = tf.reshape(annotation,(-1,))
        # labels = tf.one_hot(labels, depth=num_of_classes,dtype=tf.float32)

        # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
        #                                                                       labels=tf.squeeze(annotation,squeeze_dims=[3]),
        #                                                                       name="entropy")))


        # print('++++++++++++++++++++++++++++=logits.shape=', logits.get_shape())
        # # print('++++++++++++++++++++++++++++=labels.dtype=',labels.dtype)
        # print('++++++++++++++++++++++++++++=labels.shape()=', labels.get_shape())
        # print('------------------------positive count = ',positive_count)
        # print('------------------------negativa count = ', negativa_count)
        # print('------------------------classes_weights = ', classes_weights)
        # logits = tf.reshape(logits, (-1, num_of_classes))
        # logits = tf.nn.softmax(logits)

        # loss = tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(logits=logits,
        #                                              targets=labels,
        #                                              pos_weight=classes_weights))
        # loss = tf.reduce_mean(self.weighted_cross_entropy(targets=labels,
        #                                                   logits=logits,
        #                                                   pos_weight=classes_weights))

        loss = self.weighted_softmax_cross_entropy_loss(logits=logits,labels=labels,weights=classes_weights)

        # L2 = 0
        # for p in tl.layers.get_variables_with_name('/W', True, True):
        #     L2 += tf.contrib.layers.l2_regularizer(0.00001)(p)
        # loss = loss + L2
        return loss

    def weighted_softmax_cross_entropy_loss(self, logits, labels, weights):
        """
        Computes the SoftMax Cross Entropy loss with class weights based on the class of each pixel.

        Parameters
        ----------
        logits: TF tensor
            The network output before SoftMax.
        labels: TF tensor
            The desired output from the ground truth.
        weights : list of floats
            A list of the weights associated with the different labels in the ground truth.

        Returns
        -------
        loss : TF float
            The loss.
        weight_map: TF Tensor
            The loss weights assigned to each pixel. Same dimensions as the labels.

        """

        with tf.name_scope('loss'):
            # logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
            # labels = tf.reshape(labels, [-1], name='flatten_labels')

            weight_map = tf.to_float(tf.equal(labels, 0, name='label_map_0')) * weights[0]
            for i, weight in enumerate(weights[1:], start=1):
                weight_map = weight_map + tf.to_float(tf.equal(labels, i, name='label_map_' + str(i))) * weight

            weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

            # compute cross entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                           name='cross_entropy_softmax')

            # apply weights to cross entropy loss
            weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

            # get loss scalar
            loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

        return loss

    def weighted_softmax_cross_entropy_loss_with_false_positive_weights(self, logits, labels, weights,
                                                                        false_positive_factor=0.5):
        """
        Computes the SoftMax Cross Entropy loss with class weights based on the class of each pixel and an additional weight
        for false positive classifications (instances of class 0 classified as class 1).

        Parameters
        ----------
        logits: TF tensor
            The network output before SoftMax.
        labels: TF tensor
            The desired output from the ground truth.
        weights : list of floats
            A list of the weights associated with the different labels in the ground truth.
        false_positive_factor: float
            False positives receive a loss weight of false_positive_factor * label_weights[1], the weight of the class of interest.

        Returns
        -------
        loss : TF float
            The loss.
        weight_map: TF Tensor
            The loss weights assigned to each pixel. Same dimensions as the labels.

        """

        with tf.name_scope('loss'):
            logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
            labels = tf.reshape(labels, [-1], name='flatten_labels')

            # get predictions from likelihoods
            prediction = tf.argmax(logits, 1, name='predictions')

            # get maps of class_of_interest pixels
            prediction_map = tf.equal(prediction, 1, name='prediction_map')
            label_map = tf.equal(labels, 1, name='label_map')

            false_positive_map = tf.logical_and(prediction_map, tf.logical_not(label_map), name='false_positive_map')

            label_map = tf.to_float(label_map)
            false_positive_map = tf.to_float(false_positive_map)

            weight_map = label_map * (weights[1] - weights[0]) + weights[0]
            weight_map = tf.add(weight_map, false_positive_map * ((false_positive_factor * weights[1]) - weights[0]),
                                name="combined_weight_map")

            weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

            # compute cross entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                           name='cross_entropy_softmax')

            # apply weights to cross entropy loss
            weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

            # get loss scalar
            loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

        return loss

    def weighted_cross_entropy(self, targets, logits, pos_weight, name=None):
        """computer weight cross entropy"""
        with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
            logits = ops.convert_to_tensor(logits, name="logits")
            targets = ops.convert_to_tensor(targets, name="targets")
            try:
                targets.get_shape().merge_with(logits.get_shape())
            except ValueError:
                raise ValueError(
                    "logits and targets must have the same shape (%s vs %s)" %
                    (logits.get_shape(), targets.get_shape()))

            # log_weight = 1 + (pos_weight - 1) * targets
            # return math_ops.add(
            #     (1 - targets) * logits,
            #     log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
            #                   nn_ops.relu(-logits)),
            #     name=name)
            cross_entropy = -tf.add(targets[:, :, :, 0] * tf.log(logits[:, :, :, 0]),
                                    pos_weight*targets[:, :, :, 1] * tf.log(logits[:, :, :, 1]))
            return cross_entropy

    def train(self, loss, learning_rate, globalstep):
        # If use tf.nn.sparse_softmax_cross_entropy_with_logits ,
        # maybe loss will be NAN,because without clip
        # annotation = tf.cast(annotation,dtype = tf.float32)
        # prob = tf.nn.softmax(logits)
        # loss = -tf.reduce_mean(annotation*tf.log(tf.clip_by_value(prob,1e-11,1.0)))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        var_list = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads, global_step=globalstep)
        return train_op


# AUPR score
def computeConfMatElements(thresholded_proba_map, ground_truth):
    P = np.count_nonzero(ground_truth)
    TP = np.count_nonzero(thresholded_proba_map * ground_truth)
    FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map * ground_truth))

    return P, TP, FP


def computeAUPR(proba_map, ground_truth, threshold_list):
    proba_map = proba_map.astype(np.float32)
    proba_map = proba_map.reshape(-1)
    ground_truth = ground_truth.reshape(-1)
    precision_list_treshold = []
    recall_list_treshold = []
    # loop over thresholds
    for threshold in threshold_list:
        # threshold the proba map
        thresholded_proba_map = np.zeros(np.shape(proba_map))
        thresholded_proba_map[proba_map >= threshold] = 1
        # print(np.shape(thresholded_proba_map)) #(400,640)

        # compute P, TP, and FP for this threshold and this proba map
        P, TP, FP = computeConfMatElements(thresholded_proba_map, ground_truth)

        # check that ground truth contains at least one positive
        if (P > 0 and (TP + FP) > 0):
            precision = TP * 1. / (TP + FP)
            recall = TP * 1. / P
        else:
            precision = 1
            recall = 0

        # average sensitivity and FP over the proba map, for a given threshold
        precision_list_treshold.append(precision)
        recall_list_treshold.append(recall)

    # aupr = 0.0
    # for i in range(1, len(precision_list_treshold)):
    #     aupr = aupr + precision_list_treshold[i] * (recall_list_treshold[i] - recall_list_treshold[i - 1])
    precision_list_treshold.append(1)
    recall_list_treshold.append(0)
    return auc(recall_list_treshold, precision_list_treshold)


def main(argv=None):
    myUnet = Unet()
    image = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='image')  # input gray images
    annotation = tf.placeholder(tf.int32, shape=[None, IMG_SIZE, IMG_SIZE, 1], name="annotation")
    # image = tf.cast(image, tf.float32)
    # annotation = tf.cast(annotation, tf.int32)

    # define inferences
    train_pred, train_logits, train_tlnetwork = myUnet.model(image, is_train=True, reuse=False)
    train_positive_prob = tf.nn.softmax(train_logits)[:, :, :, 1]
    train_loss_op = myUnet.loss(train_logits, annotation)

    n_epoch = MAX_EPOCH
    n_step_epoch = int(train_nbr / batch_size)
    LR_start = learningrate
    LR_fin = learningrateend
    LR_decay = (LR_fin / LR_start) ** (1.0 / n_epoch)
    step_decay = n_step_epoch
    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learningrate, global_steps, step_decay, LR_decay, staircase=True)
    # learning_rate = tf.Variable(learningrate, dtype=tf.float32)
    # train_op = myUnet.train(train_loss_op,learning_rate,global_steps)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss_op, global_step=global_steps)
    summaries=[]
    summaries.append(tf.summary.scalar('learning_rate', learning_rate, collections=['learningrate']))
    # Merge all summaries together.
    summary_lr = tf.summary.merge(summaries, name='summary_lr')

    test_pred, test_logits, test_tlnetwork = myUnet.model(image, is_train=False, reuse=True)
    test_positive_prob = tf.nn.softmax(test_logits)[:, :, :, 1]
    test_loss_op = myUnet.loss(test_logits, annotation)

    # lr_assign_op = tf.assign(learning_rate, learning_rate / 5)  # learning_rate decay

    # only visualize the test images
    # first lighten the annotation images
    visual_annotation = tf.where(tf.equal(annotation, 1), annotation + 254, annotation)
    visual_pred = tf.expand_dims(tf.where(tf.equal(test_pred, 1), test_pred + 254, test_pred), dim=3)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(visual_annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(visual_pred, tf.uint8), max_outputs=2)

    print("Setting up summary op...")
    test_summary_op = tf.summary.merge_all()

    if FLAGS.mode == 'train':
        train_dataset_reader, validation_dataset_reader = myUnet.load_data_util()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=2)
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(logs_dir)  # if model has been trained,restore it
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    start = time.time()
    for epo in range(MAX_EPOCH):
        start_time = time.time()
        train_loss, test_loss, train_aupr, test_aupr, train_auc, test_auc = 0, 0, 0, 0, 0, 0

        for s in range(step_every_epoch):
            train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
            feed_dict = {image: train_images, annotation: train_annotations}
            tra_positive_prob, train_err, _ = sess.run([train_positive_prob, train_loss_op, train_op],
                                                       feed_dict=feed_dict)

            # compute auc score
            temp_train_annotations = np.reshape(train_annotations, -1)
            temp_tra_positive_prob = np.reshape(tra_positive_prob, -1)
            train_sauc = roc_auc_score(temp_train_annotations, np.nan_to_num(temp_tra_positive_prob))
            # compute aupr
            train_saupr = computeAUPR(np.nan_to_num(tra_positive_prob).reshape(-1), train_annotations.reshape(-1), range_threshold)

            train_loss += train_err
            train_auc += train_sauc
            train_aupr += train_saupr

        if epo + 1 == 1 or (epo + 1) % print_freq == 0:
            train_loss = train_loss / step_every_epoch
            train_auc = train_auc / step_every_epoch
            train_aupr = train_aupr / step_every_epoch
            # visualize the training loss
            print("%d epoches %d took %fs" % (print_freq, epo, time.time() - start_time))
            print("   train loss: %f" % train_loss)
            print("   train auc: %f" % train_auc)
            print("   train aupr: %f" % train_aupr)

            train_summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.Summary.Value(tag="train_aupr", simple_value=train_aupr)
            ])
            summary_writer.add_summary(train_summary, epo)
            summary_str = sess.run(summary_lr)
            summary_writer.add_summary(summary_str, epo)
            summary_writer.flush()

            for test_s in range(test_every_epoch):
                # get validation data
                valid_images, valid_annotations = validation_dataset_reader.next_batch(test_batchsize)
                # visualize the validation loss
                feed_dict = {image: valid_images, annotation: valid_annotations}
                valid_positive_prob, validation_err = sess.run([test_positive_prob, test_loss_op], feed_dict=feed_dict)
                # compute auc score
                temp_valid_annotations = np.reshape(valid_annotations, -1)
                temp_valid_positive_prob = np.reshape(valid_positive_prob, -1)
                test_sauc = roc_auc_score(temp_valid_annotations, np.nan_to_num(temp_valid_positive_prob))
                # compute test aupr
                test_saupr = computeAUPR(np.nan_to_num(valid_positive_prob).reshape(-1), valid_annotations.reshape(-1),
                                         range_threshold)

                test_loss += validation_err
                test_auc += test_sauc
                test_aupr += test_saupr
            test_loss = test_loss / test_every_epoch
            test_auc = test_auc / test_every_epoch
            test_aupr = test_aupr / test_every_epoch
            print("   test aupr: %f" % test_aupr)
            test_summary = tf.Summary(value=[
                tf.Summary.Value(tag="test_loss", simple_value=test_loss),
                tf.Summary.Value(tag="test_auc", simple_value=test_auc),
                tf.Summary.Value(tag="test_aupr", simple_value=test_aupr)
            ])
            summary_writer.add_summary(test_summary, epo)

            # visualize the test result(only visualize the last batchsize of this epoch)
            feed_dict = {image: valid_images, annotation: valid_annotations}
            summary_str = sess.run(test_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, epo)

            # tensorboard flush
            summary_writer.flush()
            sys.stdout.flush()
        # if (epo+1) % 100 == 0:
        #     sess.run(lr_assign_op)
        if (epo + 1) % 100 == 0:
            saver.save(sess, logs_dir + "model.ckpt", epo)
            print('the %d epoch , the model has been saved successfully' % epo)
            sys.stdout.flush()
    print('-------------------total cost time: %fs' % (time.time() - start))
    summary_writer.close()
    sess.close()


if __name__ == '__main__':
    tf.app.run()
