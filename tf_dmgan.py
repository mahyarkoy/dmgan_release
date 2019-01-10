#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:10:34 2017

@author: mahyar khayatkhoei
"""

import numpy as np
import tensorflow as tf
import os
import cPickle as pk

tf_dtype = tf.float32
np_dtype = 'float32'

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def dense(x, h_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
    return h1

class DMGAN:
	def __init__(self, sess, data_dim, log_dir='logs'):
		self.log_dir = log_dir
		self.sess = sess

		### optimization parameters
		self.g_lr = 2e-4
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 2e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5
		self.e_lr = 2e-4
		self.e_beta1 = 0.9
		self.e_beta2 = 0.999
		self.pg_lr = 1e-3
		self.pg_beta1 = 0.5
		self.pg_beta2 = 0.5

		### network parameters
		self.batch_size = 128
		self.z_dim = 100
		self.g_num = 10
		self.z_range = 1.0
		self.data_dim = data_dim
		self.gp_loss_weight = 10.0
		self.en_loss_weight = 1.0
		self.pg_q_lr = 0.01
		self.pg_temp = 1.0
		self.g_rl_vals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.g_rl_pvals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		self.d_act = lrelu
		self.g_act = tf.nn.relu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		### placeholders for image and label inputs
		self.im_input = tf.placeholder(tf_dtype, [None, self.data_dim], name='im_input')
		self.z_input = tf.placeholder(tf.int32, [None], name='z_input')
		self.zi_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='zi_input')
		self.e_input = tf.placeholder(tf_dtype, [None, 1], name='e_input')
		self.train_phase = tf.placeholder(tf.bool, name='phase')

		### build generator
		self.g_layer = self.build_gen(self.z_input, self.zi_input, self.g_act, self.train_phase)

		### build discriminator and encoder (Q network)
		self.r_logits, self.r_hidden = self.build_dis(self.im_input, self.d_act, self.train_phase)
		self.g_logits, self.g_hidden = self.build_dis(self.g_layer, self.d_act, self.train_phase, reuse=True)
		self.r_en_logits = self.build_encoder(self.r_hidden, self.d_act, self.train_phase)
		self.g_en_logits = self.build_encoder(self.g_hidden, self.d_act, self.train_phase, reuse=True)

		### real gen manifold interpolation (to be used in gradient penalty)
		rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
		self.rg_logits, _ = self.build_dis(rg_layer, self.d_act, self.train_phase, reuse=True)

		### build d losses
		if self.d_loss_type == 'log':
			self.d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype))
			self.d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
			self.d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.rg_logits, labels=tf.ones_like(self.rg_logits, tf_dtype))
		elif self.d_loss_type == 'was':
			self.d_r_loss = -self.r_logits 
			self.d_g_loss = self.g_logits
			self.d_rg_loss = -self.rg_logits
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### gradient penalty (one sided)
		### NaN free norm gradient
		rg_grad = tf.gradients(self.rg_logits, rg_layer)
		rg_grad_flat = tf.reshape(rg_grad, [-1, np.prod(self.data_dim)])
		rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 1.
		rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
		#rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
		rg_grad_abs = 0. * rg_grad_flat
		rg_grad_norm = tf.where(rg_grad_ok, 
			tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
		gp_loss = tf.square(rg_grad_norm - 1.0)

		### d loss combination
		self.d_loss_mean = tf.reduce_mean(self.d_r_loss + self.d_g_loss)
		self.d_loss_total = self.d_loss_mean + self.gp_loss_weight * tf.reduce_mean(gp_loss)

		### build g loss
		if self.g_loss_type == 'log':
			self.g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
		elif self.g_loss_type == 'mod':
			self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf_dtype))
		elif self.g_loss_type == 'was':
			self.g_loss = -self.g_logits
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

		self.g_loss_mean = tf.reduce_mean(self.g_loss, axis=None)
		
		### generated encoder loss: lower bound on mutual_info(z_input, generator id)
		self.g_en_loss = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			logits=self.g_en_logits)

		### discounter
		self.rl_counter = tf.get_variable('rl_counter', dtype=tf_dtype,
			initializer=1.0)

		### g loss combination
		self.g_loss_total = self.g_loss_mean + self.en_loss_weight * tf.reduce_mean(self.g_en_loss)

		### e loss combination
		self.en_loss_total = tf.reduce_mean(self.g_en_loss)

		### collect params
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
		self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")

		### compute stats of weights
		self.nan_vars = 0.
		self.inf_vars = 0.
		self.zero_vars = 0.
		self.big_vars = 0.
		self.count_vars = 0
		for v in self.g_vars + self.d_vars:
			self.nan_vars += tf.reduce_sum(tf.cast(tf.is_nan(v), tf_dtype))
			self.inf_vars += tf.reduce_sum(tf.cast(tf.is_inf(v), tf_dtype))
			self.zero_vars += tf.reduce_sum(tf.cast(tf.square(v) < 1e-6, tf_dtype))
			self.big_vars += tf.reduce_sum(tf.cast(tf.square(v) > 1.0, tf_dtype))
			self.count_vars += tf.reduce_prod(v.get_shape())
		self.count_vars = tf.cast(self.count_vars, tf_dtype)
		self.zero_vars /= self.count_vars
		self.big_vars /= self.count_vars

		self.g_vars_count = 0
		self.d_vars_count = 0
		self.e_vars_count = 0
		for v in self.g_vars:
			self.g_vars_count += int(np.prod(v.get_shape()))
		for v in self.d_vars:
			self.d_vars_count += int(np.prod(v.get_shape()))
		for v in self.e_vars:
			self.e_vars_count += int(np.prod(v.get_shape()))

		### build optimizers
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		print '>>> update_ops list: ', update_ops
		with tf.control_dependencies(update_ops):
			self.g_opt = tf.train.AdamOptimizer(
				self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
				self.g_loss_total, var_list=self.g_vars)
			self.d_opt = tf.train.AdamOptimizer(
				self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
				self.d_loss_total, var_list=self.d_vars)
			self.e_opt = tf.train.AdamOptimizer(
				self.e_lr, beta1=self.e_beta1, beta2=self.e_beta2).minimize(
				self.en_loss_total, var_list=self.e_vars)

		### summaries
		g_loss_sum = tf.summary.scalar("g_loss", self.g_loss_mean)
		d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_mean)
		e_loss_sum = tf.summary.scalar("e_loss", self.en_loss_total)
		self.summary = tf.summary.merge([g_loss_sum, d_loss_sum, e_loss_sum])

		### prior learning vars
		self.pg_var = tf.get_variable('pg_var', dtype=tf_dtype,
			initializer=self.g_rl_vals)
		self.pg_q = tf.get_variable('pg_q', dtype=tf_dtype,
			initializer=self.g_rl_vals)
		self.pg_var_flat = self.pg_temp * tf.reshape(self.pg_var, [1, -1])
		
		### prior entropy
		self.gi_h = -tf.reduce_sum(tf.nn.softmax(self.pg_var) * tf.nn.log_softmax(self.pg_var))
		
		### empirical average prior
		pg_reward = tf.reduce_mean(self.r_en_logits, axis=0)
		print '>>> pg_reward shape:', pg_reward.get_shape()
		
		### counter used for decay rate control
		rl_counter_opt = tf.assign(self.rl_counter, self.rl_counter * 0.999)
		
		### r_en_logits as q values (empirical average)
		pg_q_opt = tf.assign(self.pg_q, (1-self.pg_q_lr)*self.pg_q + \
			self.pg_q_lr * pg_reward)

		### cross entropy E_x H(p(c|x)||q(c)) - prior_entropy
		with tf.control_dependencies([pg_q_opt, rl_counter_opt]):
			en_pr = tf.nn.softmax(self.r_en_logits)
			pg_loss_total = -tf.reduce_mean(en_pr * tf.nn.log_softmax(self.pg_var)) \
				- 1000. * self.rl_counter * self.gi_h

		### prior learning optimizer
		self.pg_opt = tf.train.AdamOptimizer(
				self.pg_lr, beta1=self.pg_beta1, beta2=self.pg_beta2).minimize(
				pg_loss_total, var_list=[self.pg_var])

	def build_gen(self, z, zi, act, train_phase):
		ol = list()
		with tf.variable_scope('g_net'):
			for gi in range(self.g_num):
				with tf.variable_scope('gnum_%d' % gi):
					zi = tf.random_uniform([tf.shape(z)[0], self.z_dim], 
						minval=-self.z_range, maxval=self.z_range, dtype=tf_dtype)
					bn = tf.contrib.layers.batch_norm
			
					### fully connected from hidden z to data shape
					h1 = act(dense(zi, 128//4, scope='fc1'))
					h2 = act(dense(h1, 64//4, scope='fc2'))
					h3 = dense(h2, self.data_dim, scope='fco')
					ol.append(h3)

			z_1_hot = tf.reshape(tf.one_hot(z, self.g_num, dtype=tf_dtype), [-1, self.g_num, 1])
			z_map = tf.tile(z_1_hot, [1, 1, self.data_dim])
			os = tf.stack(ol, axis=1)
			o = tf.reduce_sum(os * z_map, axis=1)
			return o

	def build_dis(self, data_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('d_net'):
			h1 = act(dense(data_layer, 64, scope='fc1', reuse=reuse))
			h2 = act(dense(h1, 128, scope='fc2', reuse=reuse))
			o = dense(h2, 1, scope='fco', reuse=reuse)
			return o, h2

	def build_encoder(self, hidden_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('e_net'):
			with tf.variable_scope('encoder'):
				flat = hidden_layer
				flat = act(bn(dense(flat, 128, scope='fc', reuse=reuse), 
					reuse=reuse, scope='bf1', is_training=train_phase))
				o = dense(flat, self.g_num, scope='fco', reuse=reuse)
				return o

	def start_session(self):
		self.saver = tf.train.Saver(tf.global_variables(), 
			keep_checkpoint_every_n_hours=1, max_to_keep=10)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, 
		en_only=False, z_data=None, zi_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None
		
		### inf, nans, tiny and big vars stats
		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		### sample e from uniform (0,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
		e_data = e_data.astype(np_dtype)

		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### only forward encoder on batch_data
		if en_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			en_logits = self.sess.run(self.r_en_logits, feed_dict=feed_dict)
			return en_logits

		### sample zi from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
		zi_data = zi_data.astype(np_dtype)

		### multiple generator uses z_data to select gen
		self.g_rl_vals, self.g_rl_pvals = self.sess.run((self.pg_q, self.pg_var), feed_dict={})
		if z_data is None:
			g_th = self.g_num
			z_pr = np.exp(self.pg_temp * self.g_rl_pvals[:g_th])
			z_pr = z_pr / np.sum(z_pr)
			z_data = np.random.choice(g_th, size=batch_size, p=z_pr)
			#z_data = np.random.randint(low=0, high=self.g_num, size=batch_size)

		### only forward generator on z
		if gen_only:
			feed_dict = {self.z_input: z_data, self.zi_input: zi_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log
		feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.zi_input: zi_data,
			self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.summary, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.summary, self.g_opt, self.e_opt, self.pg_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)

		return res_list[1], res_list[0]
