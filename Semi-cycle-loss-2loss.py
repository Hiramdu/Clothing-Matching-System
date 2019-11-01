# coding:utf-8
import os
import numpy as np
import tensorflow as tf
import csv
import shutil
from PIL import Image
from scipy.misc import imsave
# from text_cnn import TextCNN
from datagenerator import ImageDataGenerator
# from only_textual_datagenerator import TextualDataGenerator
from datetime import datetime
import pickle
import time
import sys
from tensorflow.contrib.data import Iterator
from layers import *

from VAEmodel import *

"""
Configuration Part.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
# Path to the textfiles for the trainings and validation set
option = "/home/hanxianjing/VAE/new_ijk_data/"
train_file = option + 'train_ijk_shuffled_811.txt'
val_file = option + 'valid_ijk_shuffled_811.txt'
test_file = option + 'test_ijk_shuffled_811.txt'
# Learning params
num_epochs = 100
batch_size = 150
# How often we want to write the tf.summary data to disk
display_step = 5
input_dim = 150

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp_2loss/finetune_alexnet/tensorboard"
checkpoint_path = "tmp_2loss/tune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 shuffle=False)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  shuffle=False)
    test_data = ImageDataGenerator(test_file,
                                   mode='inference',
                                   batch_size=batch_size,
                                   shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                               tr_data.data.output_shapes)
    print("tr_data.data.output_shapes", tr_data.data.output_shapes)

    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)
test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])

outfile = "record_2loss/visual" + "batch_size_" + str(batch_size) + str(datetime.now().strftime('%H-%M-%S')) + ".txt"
# csvfile = "record2/visual"+ "batch_size_" + str(batch_size) + str(datetime.now().strftime('%H-%M-%S') )+".csv"

# def fc(x, num_in, num_out, name, relu=True):
#     """Create a fully connected layer."""
#     with tf.variable_scope(name) as scope:
#
#         # Create tf variables for the weights and biases
#         weights = tf.get_variable('weights', shape=[num_in, num_out],
#                                   trainable=False)
#         biases = tf.get_variable('biases', [num_out], trainable=False)
#
#         # Matrix multiply weights and inputs and add bias
#         act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
#     if relu:
#         # Apply ReLu non linearity
#         relu = tf.nn.relu(act)
#         return relu
#     else:
#         return act

# record by txt
file = open(outfile, "w+")
count = 0
for _learning_rate in [0.1]:
    # for _lamda in range(-2,0,1):
    for lamda in [0.001]:
        count = count + 1
        learning_rate = _learning_rate
        # lamda=_lamda
        print("lamda: {} learning_rate:{}\n".format(lamda, learning_rate))
        file.write("lamda: {} learning_rate:{}\n".format(lamda, learning_rate))
        file.write("begin time:{}".format(datetime.now()))

        input_A = x[0:batch_size:3]
        input_B = x[1:batch_size:3]
        input_B1 = x[2:batch_size:3]

        with tf.variable_scope("Model") as scope:
            

            enc_A, enc_Bg, fake_B, z_mu, z_log_sigma_sq = build_network(batch_size, input_A, "encoder_A2Bg")
            fake_rec_B, _ = build_encoder(batch_size, fake_B, "encoder_Bg")  # 生成的B走encoder网络
            enc_B, _ = build_encoder(batch_size, input_B, "encoder_B")  # B正
            scope.reuse_variables()
            enc_B1, _ = build_encoder(batch_size, input_B1, "encoder_B")  # B负

 
        enc_A_W = tf.get_variable('generator_A_W', [128, 128], dtype=tf.float32, trainable=True)
        enc_A_b = tf.get_variable('generator_A_b', [128], dtype=tf.float32, trainable=True)
        enc_B_W = tf.get_variable('generator_B_W', [128, 128], dtype=tf.float32, trainable=True)
        enc_B_b = tf.get_variable('generator_B_b', [128], dtype=tf.float32, trainable=True)
        enc_Bg_W = tf.get_variable('generator_Bg_W', [128, 128], dtype=tf.float32, trainable=True)
        enc_Bg_b = tf.get_variable('generator_Bg_b', shape=[128], dtype=tf.float32, trainable=True)

        enc_A = tf.nn.relu(tf.matmul(
            tf.reshape(tf.nn.avg_pool(value=enc_A, ksize=[1, 64, 64, 1], strides=[1, 1, 1, 1], padding='VALID'),
                       [int(batch_size / 3), -1]), enc_A_W) + enc_A_b)
        enc_B = tf.nn.relu(tf.matmul(
            tf.reshape(tf.nn.avg_pool(value=enc_B, ksize=[1, 64, 64, 1], strides=[1, 1, 1, 1], padding='VALID'),
                       [int(batch_size / 3), -1]), enc_B_W) + enc_B_b)
        enc_Bg = tf.nn.relu(tf.matmul(
            tf.reshape(tf.nn.avg_pool(value=enc_Bg, ksize=[1, 64, 64, 1], strides=[1, 1, 1, 1], padding='VALID'),
                       [int(batch_size / 3), -1]), enc_Bg_W) + enc_Bg_b)
        enc_B1 = tf.nn.relu(tf.matmul(
            tf.reshape(tf.nn.avg_pool(value=enc_B1, ksize=[1, 64, 64, 1], strides=[1, 1, 1, 1], padding='VALID'),
                       [int(batch_size / 3), -1]), enc_B_W) + enc_B_b)

        
        intra_sim_ij = tf.reduce_sum(enc_Bg*enc_B,1)
        cross_sim_ij = tf.reduce_sum(enc_A*enc_B,1)
        intra_sim_ik = tf.reduce_sum(enc_Bg*enc_B1,1)  #1*batch
        cross_sim_ik = tf.reduce_sum(enc_A*enc_B1,1)
        mij = 0.1 * intra_sim_ij + cross_sim_ij
        mik = 0.1 * intra_sim_ik + cross_sim_ik
        score = tf.subtract(mij, mik)

        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
        
            latent_loss = -0.5 * tf.reduce_sum(
                1 + z_log_sigma_sq - tf.square(z_mu) -
                tf.exp(z_log_sigma_sq), axis=1)
            latent_loss = tf.reduce_mean(latent_loss)

            

            bpr_loss = -tf.reduce_mean(tf.sigmoid(score))

            # con_loss = tf.reduce_mean(tf.abs(fake_B - input_A))
            
            con_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_B, labels = input_A), reduction_indices=1)
            con_loss = tf.reduce_mean(con_loss)

        # Train op
        with tf.name_scope("train"):
            g_optim = tf.train.AdamOptimizer(learning_rate=0.01)

            model_vars = tf.trainable_variables()
            # for var in model_vars:
            #     print(var.name)
            g_A_vars = [var for var in model_vars]

            g_grads_and_vars = g_optim.compute_gradients(latent_loss + lamda * con_loss + bpr_loss,
                                                         var_list=g_A_vars)  # 计算生成器参数梯度
            g_train = g_optim.apply_gradients(g_grads_and_vars)  # 更新生成器参数

            train_op = g_train

        # Add gradients to summary
        # g_A_loss_summ = tf.summary.scalar("g_loss", g_loss)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            res = tf.expand_dims(score, 1)
            val = tf.zeros((int(batch_size / 3), 1), tf.float32)
            res = tf.concat([res, val], 1)
            res1 = tf.argmax(res, 1)
            res2 = tf.zeros((int(batch_size / 3), 1), tf.int64)
            correct_pred = tf.equal(res1, res2)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        writer = tf.summary.FileWriter(filewriter_path)

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
        val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
        test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
        # Start Tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)
            # To continue training from one of your checkpoints
            # saver.restore(sess, "tmp_2loss/tune_alexnet/checkpoints/model_epoch17.ckpt")
            # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
            #                                                   filewriter_path))

            # Loop over number of epochs
            for epoch in range(num_epochs):
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                file.write(str(epoch + 1))
                # Initialize iterator with the training dataset
                sess.run(training_init_op)
                
                latent_loss_sum = 0.0
                bpr_loss_sum = 0.0
                con_loss_sum=0.0

                for step in range(train_batches_per_epoch):

                    img_batch = sess.run(next_batch)

                    [ms, train_op_, each_latent_loss, each_bpr_loss, each_con_loss] = sess.run([merged_summary, train_op, latent_loss, bpr_loss, con_loss], feed_dict={x: img_batch})
                    writer.add_summary(ms)
                    
                    latent_loss_sum += each_latent_loss
                    bpr_loss_sum += each_bpr_loss
                    con_loss_sum += each_con_loss

                    if step % 20 == 0:  # 每过write_pred_every次写一下训练的可视化结果
                        input_A_value, input_B_value, fake_B_value = sess.run([input_A, input_B, fake_B],
                                                                              feed_dict={x: img_batch})  # run出网络输出
                        imsave("./output/fakeB_2loss/fakeB_" + str(epoch) + "_" + str(step) + ".jpg",
                               ((fake_B_value[0] + 1) * 127.5).astype(np.float32))
                        imsave("./output/fakeB_2loss/inputA_" + str(step) + ".jpg",
                               ((input_A_value[0] + 1) * 127.5).astype(np.float32))
                        imsave("./output/fakeB_2loss/inputB_" + str(step) + ".jpg",
                               ((input_B_value[0] + 1) * 127.5).astype(np.float32))
                        #  imsave("./output/fakeB_2loss/cycA_" + str(epoch) + "_" + str(step) + ".jpg", ((cyc_A_value[0] + 1) * 127.5).astype(np.float32))

                print("con_loss_sum:{} latent_loss_sum:{} bpr_loss_sum:{}".format(con_loss_sum, latent_loss_sum,
                                                                                    bpr_loss_sum))
                file.write(
                    "con_loss_sum:{} latent_loss_sum:{} bpr_loss_sum:{}\n".format(con_loss_sum, latent_loss_sum,
                                                                                    bpr_loss_sum))
                print("{} Start Train".format(datetime.now()))
                sess.run(training_init_op)
                test_acc = 0.
                test_count = 0
                for train_epoch in range(train_batches_per_epoch):
                    img_batch = sess.run(next_batch)
                    acc = sess.run(accuracy, feed_dict={x: img_batch})

                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Train Accuracy = {:.4f}".format(datetime.now(),
                                                          test_acc))
                file.write("Train Accuracy = {:.4f}\n".format(
                    test_acc))

                sess.run(validation_init_op)
                test_acc = 0.
                test_count = 0
                for valid_epoch in range(val_batches_per_epoch):
                    img_batch = sess.run(next_batch)
                    acc = sess.run(accuracy, feed_dict={x: img_batch})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{}   Validation Accuracy = {:.4f}".format(datetime.now(),
                                                                 test_acc))
                file.write("    Validation Accuracy = {:.4f}\n".format(
                    test_acc))
                # print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                # print("{} Model checkpoint saved at {}".format(datetime.now(),
                #                                                checkpoint_name))

                sess.run(test_init_op)
                test_acc = 0.
                test_count = 0
                for test_epoch in range(test_batches_per_epoch):
                    img_batch = sess.run(next_batch)
                    acc = sess.run(accuracy, feed_dict={x: img_batch})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{}               Test Accuracy = {:.4f}".format(datetime.now(),
                                                                       test_acc))
                file.write("                Test Accuracy = {:.4f}\n".format(
                    test_acc))
                file.flush()

file.write("end time: {}".format(datetime.now()))
file.close()
