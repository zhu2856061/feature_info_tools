#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: zhuzhipeng
# @Time: 2020/1/27|10:42 下午
# @Motto： Knowledge comes from decomposition

import tensorflow as tf


class FM(object):
    def __init__(self, latent, learning_rate, l2, l1, feature_hasher_size):
        self.latent = latent
        self.lr = learning_rate
        self.l2 = l2
        self.l1 = l1
        tf.reset_default_graph()
        # placeholder
        self.x = tf.placeholder(tf.int64, [None, None])
        self.sl = tf.placeholder(tf.int64, [None, ])  # [B]
        self.y = tf.placeholder(tf.int64, [None, 1])  # [None, 2]

        # weights
        init_rand = tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=2020, dtype=tf.float32)
        self.linear_weight = tf.get_variable("linear_weight", [feature_hasher_size + 1, 1],
                                             initializer=init_rand)  # [F, 1]
        self.cross_weight = tf.get_variable("cross_weight", [feature_hasher_size + 1, latent],
                                            initializer=init_rand)  # [F, latent]

        # --------- net ---------
        # bias_part
        with tf.variable_scope("linear_bias"):
            self.bias_part = tf.get_variable("bias", [1, 1])

        # linear_part
        with tf.variable_scope("linear_part"):
            w_x = tf.nn.embedding_lookup(self.linear_weight, self.x)

            mask = tf.sequence_mask(self.sl, tf.shape(w_x)[1], dtype=tf.float32)  # [B, T]
            mask = tf.expand_dims(mask, -1)  # [B, T, 1]
            mask = tf.tile(mask, [1, 1, tf.shape(w_x)[2]])  # [B, T, H]
            w_x *= mask  # [B, T, H]

            self.linear_part = tf.reduce_sum(w_x, 1)

        # cross part
        with tf.variable_scope("cross_part"):
            v_x = tf.nn.embedding_lookup(self.cross_weight, self.x)

            mask = tf.sequence_mask(self.sl, tf.shape(v_x)[1], dtype=tf.float32)  # [B, T]
            mask = tf.expand_dims(mask, -1)  # [B, T, 1]
            mask = tf.tile(mask, [1, 1, tf.shape(v_x)[2]])  # [B, T, H]
            v_x *= mask  # [B, T, H]

            # cross sub part : sum_square part
            summed_square = tf.square(tf.reduce_sum(v_x, 1, keepdims=True))
            # cross sub part : square_sum part
            square_summed = tf.reduce_sum(tf.square(v_x), 1, keepdims=True)

            self.cross_part = tf.reduce_sum(0.5 * tf.subtract(summed_square, square_summed), axis=2)

        self.logits = self.bias_part + self.linear_part + self.cross_part
        # self.S01 = self.logits
        self.logits = tf.reshape(tf.sigmoid(self.logits), [-1, 1])
        self.y_ = self.logits
        self.logits = tf.concat([1 - self.logits, self.logits], 1)
        # self.S03 = self.logits

        self.loss = tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=self.y)

        # l2 loss
        # if self.l2 > 0:
        #     p_sum = tf.constant(0.0)
        #     for variable in tf.trainable_variables():
        #         p_sum += tf.reduce_sum(tf.nn.l2_loss(variable))
        #     self.loss += self.l2 * p_sum

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=self.l1,
            beta2=self.l2,
        ).minimize(self.loss, global_step=self.global_step)
        # trainable_params = tf.trainable_variables()
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        # gradients = tf.gradients(self.loss, trainable_params)
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        #
        #
        # self.train_op = self.opt.apply_gradients(
        #     zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, x, sl, y):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.x: x,
            self.sl: sl,
            self.y: y,
        })
        return loss

    def test(self, sess, x, sl, y):
        return sess.run([self.loss, self.y_], feed_dict={
            self.x: x,
            self.sl: sl,
            self.y: y,
        })

    @staticmethod
    def save(sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    @staticmethod
    def restore(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


# if __name__ == '__main__':
#     fm = FM(5, 0.005, 0.01, 0.99, 10)
#     x = [[1, 2, 0], [2, 3, 4]]
#     y = [[1], [0]]
#     sl = [2, 3]
#     tf.set_random_seed(2020)
#     sess = tf.InteractiveSession()
#     sess.run(tf.global_variables_initializer())
#     [s1, s2, s3, loss] = sess.run([fm.S01, fm.S02, fm.S03, fm.loss], feed_dict={
#         fm.x: x,
#         fm.sl: sl,
#         fm.y: y,
#     })
#     print(s1)
#     print(s2)
#     print(s3)
#     print(loss)
