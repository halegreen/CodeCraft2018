# coding=utf-8
import os
import pickle
import math
import time
import random
import decimal
import conf
from operator import mul
from abc import ABCMeta, abstractmethod
from data_helper import DataHelper
from data_helper import Numpy 

random.seed(0)

class LogistModel(object):
	_model_param = {'weights' : list()}

	@classmethod
	def _sigmoid(cls, X):
		return 1.0 / (1 + math.exp(-X))

	@classmethod
	def _dot_multiply(cls, l_x, l_y):
		return sum([a * b for a, b in zip(l_x, l_y)])

	@classmethod
	def train(cls, vm, train_x, train_y, save_model=False):
		'''
		 Logist:
		 train_x[i][t] = w[t1] * train_x[i][t1] + w[t2] * train_x[i][t2] + ... + w[tn] + train_x[i][tn]
		 这里的时间粒度暂时选取每台虚拟机的前面n个时刻的使用情况作为特征
		'''
		model_path = conf.model_param_path +'LogistModel_%s' % vm
		if os.path.exists(model_path):
			return model_path
		print '==========Training...%s===========' % vm
		feat_dim = max([len(x) for x in train_x])
		print 'sample_num: %s, feat_dim: %s' % (len(train_x), feat_dim)
		for x in train_x:
			if len(x) != feat_dim:
				x.extend([0 for i in range(feat_dim - len(x))])
		print train_x
		print train_y
		weights = [1.0 for i in range(feat_dim)]
		for k in range(conf.iterations_num):
			if conf.opimize_type == 'stoc_grad_descent':
				loss = 0.0 
				for i in range(len(train_x)):
					output = cls._sigmoid(cls._dot_multiply(train_x[i], weights))
					# print output, train_y[i]
					loss += 1.0 * abs(train_y[i][0] - output)
					weights = [weight + conf.alpha * x * loss for weight, x in zip(weights, train_x[i])]
				loss = 1.0 * loss / float(len(train_x))
				print '%sth iter train loss: %s' % (k, loss)
		cls._model_param['weights'] = weights
		if not os.path.exists(conf.model_param_path):
			os.makedirs(conf.model_param_path)
		if save_model:
			with open(model_path, 'w') as f:
				for w in weights:
					f.write(str(w) + '\t')
		return model_path


	@classmethod
	def predict(cls, vm, model_path, total_train_x, use_history_mean=False):
		with open(model_path, 'r') as f:
			weights = filter(lambda x: len(x) > 0,  f.read().split('\t'))
			weights = map(lambda x: float(x[0]), weights)
		week_idx = len(total_train_x) - 1
		while week_idx >= 0:
			find = False
			for item in total_train_x[week_idx]:
				if item[0] == vm:
					vm_history_info = item[1]
					find = True
					break
			if find:
				break
			week_idx -= 1
		if use_history_mean:
			non_zero_len = len(filter(lambda x: x != 0, vm_history_info))
			if non_zero_len == 0:
				return 0
			pred_res = int(reduce(lambda x, y: x + y, vm_history_info) / non_zero_len)
		else:
			pred_res = int(sum([float(weight) * int(vm_num) for weight, vm_num in zip(weights, vm_history_info)]))
			pred_res = cls._trick_fit(pred_res)
		# print vm, vm_history_info, pred_res
		return pred_res

	@classmethod
	def _trick_fit(cls, pred_num):
			if pred_num >= 40:
				return pred_num - 30
			elif pred_num >= 30:
				return pred_num - 20
			elif pred_num >= 20:
				return pred_num - 10
			elif pred_num >= 10:
				return pred_num - 5
			else:
				return pred_num

	@classmethod
	def eval_on_test_data(cls, pred_res, true_y):
		loss = 0.0
		for vm in pred_res.keys():
			loss += abs(pred_res[vm] - true_y[vm])
		print 'test loss: %s' % (1.0 * loss / float(len(pred_res)))

## NN 简单的神经网络模型 ===============================================================

class NNModel(object):
	_ni = 1

	_model_param = {'input_n' : 0,
					'hidden_n' : 0,
					'output_n' : 0,
					'input_cells' : list(),
					'hidden_cells' : list(),
					'output_cells' : list(),
					'input_weights' : list(),
					'output_weights' : list(),
					'inout_correction' : list(),
					'output_correction' : list()
				}
	@classmethod
	def _make_matrix(cls, m, n, fill=0.0):
		mat = []
		for i in range(m):
			mat.append([fill] * n)
		return mat

	@classmethod
	def _sigmoid_derivative(cls, x):
		return x * (1 - x)

	@classmethod
	def _sigmoid(cls, x):
		if x < 0:
			return 1 - 1 / (1 + math.exp(x))
		else:
			return 1 / (1 + math.exp(-x))

	@classmethod
	def _relu(cls, x):
		return max(0, x)

	@classmethod
	def _relu_derivative(cls, x):
		if x > 0:
			return 1
		else:
			return 0


	@classmethod
	def _rand(cls):
		# return random.gauss(0, 0.01)
		return random.normalvariate(0, 1)

	@classmethod
	def _set_up(cls, ni, nh, no):
		cls._model_param['input_n'] = ni + 1
		cls._model_param['hidden_n'] = nh
		cls._model_param['output_n'] = no
		cls._model_param['input_cells'] = [1.0] * cls._model_param['input_n']
		cls._model_param['hidden_cells'] = [1.0] * cls._model_param['hidden_n']
		cls._model_param['output_cells'] = [1.0] * cls._model_param['output_n']
		cls._model_param['input_weights'] = cls._make_matrix(cls._model_param['input_n'], cls._model_param['hidden_n'])
		cls._model_param['output_weights'] = cls._make_matrix(cls._model_param['hidden_n'], cls._model_param['output_n'])
		for i in range(cls._model_param['input_n']):
			for h in range(cls._model_param['hidden_n']):
				cls._model_param['input_weights'][i][h] = cls._rand()
		for h in range(cls._model_param['hidden_n']):
			for o in range(cls._model_param['output_n']):
				cls._model_param['output_weights'][h][o] = cls._rand()
		cls._model_param['input_correction'] = cls._make_matrix(cls._model_param['input_n'], cls._model_param['hidden_n'])
		cls._model_param['output_correction'] = cls._make_matrix(cls._model_param['hidden_n'], cls._model_param['output_n'])

	@classmethod
	def _back_propagate(cls, case, label, learn, correct):
		cls._inference(case, cls._model_param)
		output_deltas = [0.0] * cls._model_param['output_n']
		for o in range(cls._model_param['output_n']):
			error = label[o] - cls._model_param['output_cells'][o]
			# output_deltas[o] = cls._sigmoid_derivative(cls._model_param['output_cells'][o]) * error
			output_deltas[o] = error
		# get hidden layer error
		hidden_deltas = [0.0] * cls._model_param['hidden_n']
		for h in range(cls._model_param['hidden_n']):
			error = 0.0
			for o in range(cls._model_param['output_n']):
				error += output_deltas[o] * cls._model_param['output_weights'][h][o]
			hidden_deltas[h] = cls._sigmoid_derivative(cls._model_param['hidden_cells'][h]) * error
		# update output weights
		for h in range(cls._model_param['hidden_n']):
			for o in range(cls._model_param['output_n']):
				change = output_deltas[o] * cls._model_param['hidden_cells'][h]
				cls._model_param['output_weights'][h][o] += learn * change + correct * cls._model_param['output_correction'][h][o]
				cls._model_param['output_correction'][h][o] = change
		# update input weights
		for i in range(cls._model_param['input_n']):
			for h in range(cls._model_param['hidden_n']):
				change = hidden_deltas[h] * cls._model_param['input_cells'][i]
				cls._model_param['input_weights'][i][h] += learn * change + correct * cls._model_param['input_correction'][i][h]
				cls._model_param['input_correction'][i][h] = change
		# error = 0.0
		# for o in range(len(label)):
		# 	error += 0.5 * (label[o] - cls._model_param['output_cells'][o]) ** 2
		# return error


	@classmethod
	def train(cls, vm, train_x, train_y, save_model=False):
		losses = list()
		model_path = conf.model_param_path +'NNModel_%s' % vm
		if os.path.exists(model_path):
			return model_path
		cls._ni = len(train_x[0])
		cls._set_up(cls._ni, conf.NN_nh, conf.NN_no)
		for k in range(conf.iterations_num):
			if len(losses) > 2 and abs(losses[-1] - losses[-2]) < 1:
				break
			loss = 0.0
			for i in range(len(train_x)):
				# loss += cls._back_propagate(train_x[i], train_y[i], conf.learning_rate, conf.correct)
				cls._back_propagate(train_x[i], train_y[i], conf.learning_rate, conf.correct)
			print '%sth iter, train loss: %s' % (k, 1.0 * loss / len(train_x))
			losses.append(loss)
		if save_model:
			if not os.path.exists(conf.model_param_path):
				os.makedirs(conf.model_param_path)
			with open(model_path, 'w') as f:
				pickle.dump(cls._model_param, f)
		return model_path

	@classmethod
	def _inference(cls, inputs, model_param):
		for i in range(model_param['input_n'] - 1):
			model_param['input_cells'][i] = inputs[i]
		# activate hidden layer
		for j in range(model_param['hidden_n']):
			total = 0.0
			for i in range(model_param['input_n']):
				total += model_param['input_cells'][i] * model_param['input_weights'][i][j]
			model_param['hidden_cells'][j] = cls._sigmoid(total)
		# activate output layer
		for k in range(model_param['output_n']):
			total = 0.0
			for j in range(model_param['hidden_n']):
				total += model_param['hidden_cells'][j] * model_param['output_weights'][j][k]
			# model_param['output_cells'][k] = cls._sigmoid(total)   
			model_param['output_cells'][k] = total    ### 用做回归，最后一层不激活直接输出
		return model_param['output_cells'][:]

	@classmethod
	def predict(cls, model_path, test_x):
		with open(model_path, 'r') as f:
			model_param = pickle.load(f)
		pred = cls._inference(test_x, model_param)
		pred = int(pred[0])
		if pred < 0:
			pred = 0
		return pred



### HMM 隐马尔可夫模型 ===============================================================
class _BaseHMM():
	"""
	基本HMM虚类，需要重写关于发射概率的相关虚函数
	n_state : 隐藏状态的数目
	n_iter : 迭代次数
	x_size : 观测值维度
	start_prob : 初始概率
	transmat_prob : 状态转换概率
	"""
	__metaclass__ = ABCMeta  # 虚类声明

	def __init__(self, n_state=1, x_size=1, iter=20):
		self.n_state = n_state
		self.x_size = x_size
		self.start_prob = Numpy(None).ones(n_state) * (1.0 / n_state) # 初始状态概率
		self.transmat_prob = Numpy(None).ones((n_state, n_state)) * (1.0 / n_state)  # 状态转换概率矩阵
		self.trained = False # 是否需要重新训练
		self.n_iter = iter  # EM训练的迭代次数

	# 初始化发射参数
	@abstractmethod
	def _init(self,X):
		pass

	# 虚函数：返回发射概率
	@abstractmethod
	def emit_prob(self, x):  # 求x在状态k下的发射概率 P(X|Z)
		return Numpy(None)

	# 虚函数
	@abstractmethod
	def generate_x(self, z): # 根据隐状态生成观测值x p(x|z)
		return Numpy(None)

	# 虚函数：发射概率的更新
	@abstractmethod
	def emit_prob_updated(self, X, post_state):
		pass

	# 通过HMM生成序列
	def generate_seq(self, seq_length):
		X = np.zeros((seq_length, self.x_size))
		Z = np.zeros(seq_length)
		Z_pre = np.random.choice(self.n_state, 1, p=self.start_prob)  # 采样初始状态
		X[0] = self.generate_x(Z_pre) # 采样得到序列第一个值
		Z[0] = Z_pre

		for i in range(seq_length):
			if i == 0: continue
			# P(Zn+1)=P(Zn+1|Zn)P(Zn)
			Z_next = np.random.choice(self.n_state, 1, p=self.transmat_prob[Z_pre,:][0])
			Z_pre = Z_next
			# P(Xn+1|Zn+1)
			X[i] = self.generate_x(Z_pre)
			Z[i] = Z_pre

		return X,Z

	# 估计序列X出现的概率
	def X_prob(self, X, Z_seq=Numpy(None)):
		# 状态序列预处理
		# 判断是否已知隐藏状态
		X_length = len(X)
		if Z_seq.any():
			Z = np.zeros((X_length, self.n_state))
			for i in range(X_length):
				Z[i][int(Z_seq[i])] = 1
		else:
			Z = Numpy(None).ones((X_length, self.n_state))
		# 向前向后传递因子
		_, c = self.forward(X, Z)  # P(x,z)
		# 序列的出现概率估计
		prob_X = np.sum(np.log(c))  # P(X)
		return prob_X

	# 已知当前序列预测未来（下一个）观测值的概率
	def predict(self, X, x_next, Z_seq=Numpy(None), istrain=True):
		if self.trained == False or istrain == False:  # 需要根据该序列重新训练
			self.train(X)

		X_length = len(X)
		if Z_seq.any():
			Z = Numpy(None).zeros((X_length, self.n_state))
			for i in range(X_length):
				Z[i][int(Z_seq[i])] = 1
		else:
			Z = Numpy(None).ones((X_length, self.n_state))
		# 向前向后传递因子
		alpha, _ = self.forward(X, Z)  # P(x,z)
		prob_x_next = self.emit_prob(np.array([x_next]))*np.dot(alpha[X_length - 1],self.transmat_prob)
		return prob_x_next

	def decode(self, X, istrain=True):
		"""
		利用维特比算法，已知序列求其隐藏状态值
		:param X: 观测值序列
		:param istrain: 是否根据该序列进行训练
		:return: 隐藏状态序列
		"""
		if self.trained == False or istrain == False:  # 需要根据该序列重新训练
			self.train(X)

		X_length = len(X)  # 序列长度
		state = np.zeros(X_length)  # 隐藏状态

		pre_state = np.zeros((X_length, self.n_state))  # 保存转换到当前隐藏状态的最可能的前一状态
		max_pro_state = np.zeros((X_length, self.n_state))  # 保存传递到序列某位置当前状态的最大概率

		_,c=self.forward(X, Numpy(None).ones((X_length, self.n_state)))
		max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1/c[0]) # 初始概率

		# 前向过程
		for i in range(X_length):
			if i == 0: continue
			for k in range(self.n_state):
				prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:,k] * max_pro_state[i-1]
				max_pro_state[i][k] = np.max(prob_state)* (1/c[i])
				pre_state[i][k] = np.argmax(prob_state)

		# 后向过程
		state[X_length - 1] = np.argmax(max_pro_state[X_length - 1,:])
		for i in reversed(range(X_length)):
			if i == X_length - 1: continue
			state[i] = pre_state[i + 1][int(state[i + 1])]

		return  state

	# 针对于多个序列的训练问题
	def train_batch(self, X, Z_seq=list()):
		# 针对于多个序列的训练问题，其实最简单的方法是将多个序列合并成一个序列，而唯一需要调整的是初始状态概率
		# 输入X类型：list(array)，数组链表的形式
		# 输入Z类型: list(array)，数组链表的形式，默认为空列表（即未知隐状态情况）
		self.trained = True
		X_num = len(X) # 序列个数
		self._init(self.expand_list(X)) # 发射概率的初始化

		# 状态序列预处理，将单个状态转换为1-to-k的形式
		# 判断是否已知隐藏状态
		if Z_seq==list():
			Z = []  # 初始化状态序列list
			for n in range(X_num):
				Z.append(list(Numpy(None).ones((len(X[n]), self.n_state))))
		else:
			Z = []
			for n in range(X_num):
				Z.append(np.zeros((len(X[n]),self.n_state)))
				for i in range(len(Z[n])):
					Z[n][i][int(Z_seq[n][i])] = 1

		for e in range(self.n_iter):  # EM步骤迭代
			# 更新初始概率过程
			#  E步骤
			print "iter: ", e
			b_post_state = []  # 批量累积：状态的后验概率，类型list(array)
			b_post_adj_state = np.zeros((self.n_state, self.n_state)) # 批量累积：相邻状态的联合后验概率，数组
			b_start_prob = np.zeros(self.n_state) # 批量累积初始概率
			for n in range(X_num): # 对于每个序列的处理
				X_length = len(X[n])
				alpha, c = self.forward(X[n], Z[n])  # P(x,z)
				beta = self.backward(X[n], Z[n], c)  # P(x|z)

				post_state = alpha * beta / np.sum(alpha * beta) # 归一化！
				b_post_state.append(post_state)
				post_adj_state = np.zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
				for i in range(X_length):
					if i == 0: continue
					if c[i]==0: continue
					post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
															beta[i] * self.emit_prob(X[n][i])) * self.transmat_prob

				if np.sum(post_adj_state)!=0:
					post_adj_state = post_adj_state/np.sum(post_adj_state)  # 归一化！
				b_post_adj_state += post_adj_state  # 批量累积：状态的后验概率
				b_start_prob += b_post_state[n][0] # 批量累积初始概率

			# M步骤，估计参数，最好不要让初始概率都为0出现，这会导致alpha也为0
			b_start_prob += 0.001*Numpy(None).ones(self.n_state)
			self.start_prob = b_start_prob / np.sum(b_start_prob)
			b_post_adj_state += 0.001
			for k in range(self.n_state):
				if np.sum(b_post_adj_state[k])==0: continue
				self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

			self.emit_prob_updated(self.expand_list(X), self.expand_list(b_post_state))

	def expand_list(self, X):
		# 将list(array)类型的数据展开成array类型
		C = []
		for i in range(len(X)):
			C += list(X[i])
		return np.array(C)

	# 针对于单个长序列的训练
	def train(self, vm, X, Z_seq=Numpy(None), save_model=False):
		# 输入X类型：array，数组的形式
		# 输入Z类型: array，一维数组的形式，默认为空列表（即未知隐状态情况）
		self.trained = True
		X_length = len(X)
		self._init(X)

		# 状态序列预处理
		# 判断是否已知隐藏状态
		if Z_seq.any():
			Z = Numpy(None).zeros((X_length, self.n_state))
			for i in range(X_length):
				Z[i][int(Z_seq[i])] = 1
		else:
			Z = Numpy(None).ones((X_length, self.n_state))

		for e in range(self.n_iter):  # EM步骤迭代
			# 中间参数
			print e, " iter"
			# E步骤
			# 向前向后传递因子
			alpha, c = self.forward(X, Z)  # P(x,z)
			beta = self.backward(X, Z, c)  # P(x|z)

			post_state = alpha * beta
			post_adj_state = Numpy(None).zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
			for i in range(X_length):
				if i == 0: continue
				if c[i]==0: continue
				post_adj_state += (1 / c[i])*np.outer(alpha[i - 1],beta[i]*self.emit_prob(X[i]))*self.transmat_prob

			# M步骤，估计参数
			self.start_prob = post_state[0] / np.sum(post_state[0])
			for k in range(self.n_state):
				self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

			self.emit_prob_updated(X, post_state)

	# 求向前传递因子
	def forward(self, X, Z):
		X_length = len(X)
		alpha = Numpy(None).zeros((X_length, self.n_state))  # P(x,z)
		alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0] # 初始值
		# 归一化因子
		c = Numpy(None).zeros(X_length)
		c[0] = np.sum(alpha[0])
		alpha[0] = alpha[0] / c[0]
		# 递归传递
		for i in range(X_length):
			if i == 0: continue
			alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
			c[i] = np.sum(alpha[i])
			if c[i]==0: continue
			alpha[i] = alpha[i] / c[i]

		return alpha, c

	# 求向后传递因子
	def backward(self, X, Z, c):
		X_length = len(X)
		beta = Numpy(None).zeros((X_length, self.n_state))  # P(x|z)
		beta[X_length - 1] = Numpy(None).ones((self.n_state))
		# 递归传递
		for i in reversed(range(X_length)):
			if i == X_length - 1: continue
			beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
			if c[i+1]==0: continue
			beta[i] = beta[i] / c[i + 1]

		return beta


class HMM(_BaseHMM):
	def __init__(self, n_state=1, x_num=1, iter=20):
		_BaseHMM.__init__(self, n_state=n_state, x_size=1, iter=iter)
		self.emission_prob = Numpy(None).ones((n_state, x_num)) * (1.0/x_num)  # 初始化发射概率均值
		self.x_num = x_num

	def _init(self, X):
		self.emission_prob = Numpy(None).random((self.n_state,self.x_num))
		for k in range(self.n_state):
			self.emission_prob[k] = (self.emission_prob[k] / Numpy(None).sum(self.emission_prob[k])).to_list()

	def emit_prob(self, x): # 求x在状态k下的发射概率
		prob = Numpy(None).zeros(self.n_state)
		for i in range(self.n_state): prob[i]=self.emission_prob[i][int(x[0])]
		return prob

	def generate_x(self, z): # 根据状态生成x p(x|z)
		return np.random.choice(self.x_num, 1, p=self.emission_prob[z][0])

	def emit_prob_updated(self, X, post_state): # 更新发射概率
		self.emission_prob = Numpy(None).zeros((self.n_state, self.x_num))
		X_length = len(X)
		for n in range(X_length):
			self.emission_prob[:,int(X[n])] += post_state[n]

		self.emission_prob+= 0.1/self.x_num
		for k in range(self.n_state):
			if np.sum(post_state[:,k])==0: continue
			self.emission_prob[k] = self.emission_prob[k] / np.sum(post_state[:,k])

	

### LSTM ==============================================================

class LstmParam:
	def __init__(self, mem_cell_ct, x_dim):
		self.mem_cell_ct = mem_cell_ct
		self.x_dim = x_dim
		concat_len = x_dim + mem_cell_ct
		# weight matrices
		self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
		self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
		# bias terms
		self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
		self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
		# diffs (derivative of loss function w.r.t. all parameters)
		self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
		self.bg_diff = np.zeros(mem_cell_ct) 
		self.bi_diff = np.zeros(mem_cell_ct) 
		self.bf_diff = np.zeros(mem_cell_ct) 
		self.bo_diff = np.zeros(mem_cell_ct) 

	def apply_diff(self, lr = 1):
		self.wg -= lr * self.wg_diff
		self.wi -= lr * self.wi_diff
		self.wf -= lr * self.wf_diff
		self.wo -= lr * self.wo_diff
		self.bg -= lr * self.bg_diff
		self.bi -= lr * self.bi_diff
		self.bf -= lr * self.bf_diff
		self.bo -= lr * self.bo_diff
		# reset diffs to zero
		self.wg_diff = np.zeros_like(self.wg)
		self.wi_diff = np.zeros_like(self.wi) 
		self.wf_diff = np.zeros_like(self.wf) 
		self.wo_diff = np.zeros_like(self.wo) 
		self.bg_diff = np.zeros_like(self.bg)
		self.bi_diff = np.zeros_like(self.bi) 
		self.bf_diff = np.zeros_like(self.bf) 
		self.bo_diff = np.zeros_like(self.bo) 

class LstmState:
	def __init__(self, mem_cell_ct, x_dim):
		self.g = np.zeros(mem_cell_ct)
		self.i = np.zeros(mem_cell_ct)
		self.f = np.zeros(mem_cell_ct)
		self.o = np.zeros(mem_cell_ct)
		self.s = np.zeros(mem_cell_ct)
		self.h = np.zeros(mem_cell_ct)
		self.bottom_diff_h = np.zeros_like(self.h)
		self.bottom_diff_s = np.zeros_like(self.s)
	
class LstmNode:
	def __init__(self, lstm_param, lstm_state):
		# store reference to parameters and to activations
		self.state = lstm_state
		self.param = lstm_param
		# non-recurrent input concatenated with recurrent input
		self.xc = None

	def bottom_data_is(self, x, s_prev = None, h_prev = None):
		# if this is the first lstm node in the network
		if s_prev is None: s_prev = np.zeros_like(self.state.s)
		if h_prev is None: h_prev = np.zeros_like(self.state.h)
		# save data for use in backprop
		self.s_prev = s_prev
		self.h_prev = h_prev

		# concatenate x(t) and h(t-1)
		xc = np.hstack((x,  h_prev))
		self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
		self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
		self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
		self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
		self.state.s = self.state.g * self.state.i + s_prev * self.state.f
		self.state.h = self.state.s * self.state.o

		self.xc = xc
	
	def top_diff_is(self, top_diff_h, top_diff_s):
		# notice that top_diff_s is carried along the constant error carousel
		ds = self.state.o * top_diff_h + top_diff_s
		do = self.state.s * top_diff_h
		di = self.state.g * ds
		dg = self.state.i * ds
		df = self.s_prev * ds

		# diffs w.r.t. vector inside sigma / tanh function
		di_input = sigmoid_derivative(self.state.i) * di 
		df_input = sigmoid_derivative(self.state.f) * df 
		do_input = sigmoid_derivative(self.state.o) * do 
		dg_input = tanh_derivative(self.state.g) * dg

		# diffs w.r.t. inputs
		self.param.wi_diff += np.outer(di_input, self.xc)
		self.param.wf_diff += np.outer(df_input, self.xc)
		self.param.wo_diff += np.outer(do_input, self.xc)
		self.param.wg_diff += np.outer(dg_input, self.xc)
		self.param.bi_diff += di_input
		self.param.bf_diff += df_input       
		self.param.bo_diff += do_input
		self.param.bg_diff += dg_input       

		# compute bottom diff
		dxc = np.zeros_like(self.xc)
		dxc += np.dot(self.param.wi.T, di_input)
		dxc += np.dot(self.param.wf.T, df_input)
		dxc += np.dot(self.param.wo.T, do_input)
		dxc += np.dot(self.param.wg.T, dg_input)

		# save bottom diffs
		self.state.bottom_diff_s = ds * self.state.f
		self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
	def __init__(self, lstm_param):
		self.lstm_param = lstm_param
		self.lstm_node_list = []
		# input sequence
		self.x_list = []

	def y_list_is(self, y_list, loss_layer):
		"""
		Updates diffs by setting target sequence 
		with corresponding loss layer. 
		Will *NOT* update parameters.  To update parameters,
		call self.lstm_param.apply_diff()
		"""
		assert len(y_list) == len(self.x_list)
		idx = len(self.x_list) - 1
		# first node only gets diffs from label ...
		loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
		diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
		# here s is not affecting loss due to h(t+1), hence we set equal to zero
		diff_s = np.zeros(self.lstm_param.mem_cell_ct)
		self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
		idx -= 1

		### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
		### we also propagate error along constant error carousel using diff_s
		while idx >= 0:
			loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
			diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
			diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
			diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
			self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
			idx -= 1 

		return loss

	def x_list_clear(self):
		self.x_list = []

	def x_list_add(self, x):
		self.x_list.append(x)
		if len(self.x_list) > len(self.lstm_node_list):
			# need to add new lstm node, create new state mem
			lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
			self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

		# get index of most recent x input
		idx = len(self.x_list) - 1
		if idx == 0:
			# no recurrent inputs yet
			self.lstm_node_list[idx].bottom_data_is(x)
		else:
			s_prev = self.lstm_node_list[idx - 1].state.s
			h_prev = self.lstm_node_list[idx - 1].state.h
			self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

class ToyLossLayer:
	"""
	Computes square loss with first element of hidden layer array.
	"""
	@classmethod
	def loss(self, pred, label):
		return (pred[0] - label) ** 2

	@classmethod
	def bottom_diff(self, pred, label):
		diff = np.zeros_like(pred)
		diff[0] = 2 * (pred[0] - label)
		return diff




