# coding=utf-8
import types
import random
import datetime as date
from operator import mul, add
from collections import Counter

class DataHelper(object):

	_local_time_infos = {}

	@classmethod
	def split_data(cls, original_data, split_interval='week'):
		if split_interval == 'week':
			### 统计每个星期的各个虚拟机的数量
			time_range = {'year' : list(), 'month' : list()}
			datetime_list = map(lambda x: 
				date.datetime.strptime(x.split('\t')[2].strip(), "%Y-%m-%d %H:%M:%S"), original_data)
			year_set = set(map(lambda x: x.year, datetime_list))
			min_year = min(year_set); max_year = max(year_set)
			cls._local_time_infos['max_year'] = int(max_year)
			# print 'year range: %s ~ %s' %(str(min_year), str(max_year))
			if len(year_set) == 1:
				month_set = set(map(lambda x: x.month, datetime_list))
				min_month = min(list(month_set)); max_month = max(list(month_set))
				cls._local_time_infos['max_month'] = int(max_month)
				cls._local_time_infos['min_month'] = int(min_month)
				# print 'month range: %s ~ %s' % (str(min_month), str(max_month))
				time_range['year'].append(min_year)
				for i in range(min_month, max_month + 1):
					time_range['month'].append(i)
			time_info = list()  ### 机器的时间信息统计
			for month in time_range['month']:
				start_day = min(map(lambda x: x.day, 
				filter(lambda x: x.month == month, datetime_list)))
				cur_month_day = map(lambda x: x.day, filter(lambda x: x.month == month, datetime_list))
				while start_day < max(cur_month_day) and start_day < 31:
					end_day = min(start_day + 7, max(cur_month_day))   ### 6 or 7
					# print month, start_day, end_day
					# print 'month: %s, start_day: %s, end_day: %s' %(month, start_day, end_day)
					cur_day_range = filter(lambda x: 
						x.day >= start_day and x.day <= end_day, filter(lambda x: x.month == month, datetime_list))
					cur_day_vm = filter(lambda x: 
						date.datetime.strptime(x.split('\t')[2].strip(), "%Y-%m-%d %H:%M:%S") in cur_day_range, original_data)
					t_cur_day_vm = map(lambda x: x.split('\t')[1].strip(), cur_day_vm)
					tmp_cnt = Counter(t_cur_day_vm)
					# for i in range(5):
					# 	if 'flavor%s' % (str(i + 1)) in tmp_cnt.keys():
					# 		print 'flavor%s' % (str(i + 1)), tmp_cnt['flavor%s' % (str(i + 1))]
					time_info.append({'week_cnt' : tmp_cnt, 'month_startday' : str(month) + '_' + str(start_day)})
					start_day = end_day + 1
			# print time_info
			return time_info, datetime_list

	@classmethod
	def trans_data_format(cls, time_info):
		all_vm_flavor = set()
		batch_num = len(time_info)
		for i in range(batch_num):
			all_vm_flavor.update(time_info[i]['week_cnt'].keys())
		train_x = list(); train_y = list()
		for week_idx in range(1, batch_num):   ### 从第1周开始算，第0周由于没有历史信息
			t_x = list(); t_y = list()
			# print week_idx
			for vm_flavor in all_vm_flavor:
				if vm_flavor not in time_info[week_idx]['week_cnt'].keys():
					continue
				### TODO:每种机型上几周的使用情况: 第一周数据不考虑，从第二周开始预测
				former_time_info = cls._find_former_time_info(vm_flavor,time_info, week_idx)
				# print vm_flavor, former_time_info
				t_x.append((vm_flavor, former_time_info))  
				t_y.append(time_info[week_idx]['week_cnt'][vm_flavor])
			train_x.append(t_x)
			train_y.append(t_y)
		return train_x, train_y

	@classmethod
	def _find_former_time_info(cls, vm_flavor, time_info, week_idx):
		'''
		某规格机器在前n个时刻(前n周）的信息:
		train_x[i][t] = w[t1] * train_x[i][t1] + w[t2] * train_x[i][t2] + ... + w[tn] + train_x[i][tn]
		'''
		month, day = time_info[week_idx]['month_startday'].split('_')
		former_time_info = list()
		for j in range(len(time_info)):
			if vm_flavor in time_info[j]['week_cnt'].keys():
				f_month, f_startday = time_info[j]['month_startday'].split('_')
				break
		if f_month == month and f_startday == day:
			while week_idx > 0:
				former_time_info.append(0)
				week_idx -= 1
		else:
			for cur_week in range(0, week_idx):
				if vm_flavor not in time_info[cur_week]['week_cnt'].keys():
					former_time_info.append(0)
				else:
					former_time_info.append(time_info[cur_week]['week_cnt'][vm_flavor])
		return former_time_info

	@classmethod
	def sliding_window(cls, t_train_x, t_train_y, vm):
		train_x = []
		train_y = []
		week_num = len(t_train_x)
		if week_num > 6:
			interval = 6
		elif week_num >= 4:
			interval = 2
		else:
			interval = 1
		total_history_info = cls._data_preprocess(t_train_x[week_num - 1])
		start_day = 0; end_day = start_day + interval   ### 前6周预测后一周
		while end_day + 1 < len(total_history_info):
			t_x = total_history_info[start_day: end_day]
			t_y = [total_history_info[end_day]]
			train_x.append(t_x)
			train_y.append(t_y)
			start_day = start_day + 1
			end_day = start_day + interval
		test_x = total_history_info[start_day: end_day]
		test_y = [total_history_info[end_day]]
		return train_x, train_y, test_x, test_y


	@classmethod
	def get_need_predict(cls, input_lines):
		ret_vm = list()
		for i in range(3, len(input_lines)):
			line = input_lines[i].strip()
			if len(line) == 0 or line.strip() == 'CPU' or line.strip() == 'MEM':
				break
			ret_vm.append(line.split()[0])
		return ret_vm

	@classmethod
	def get_specific_vm(cls, total_train_x, total_train_y, vm):
		train_x = list(); train_y = list()
		for week_idx in range(len(total_train_x)):
			for x, y in zip(total_train_x[week_idx], total_train_y[week_idx]):
				if x[0] == vm:
					train_x.append(x[1])
					train_y.append(y)
		return train_x, train_y

	@classmethod
	def write_final_res(cls, pred_res, dis_res):
		result = list()
		total_vm_num = 0
		for k in pred_res.keys():
			total_vm_num += pred_res[k]
		total_server_num = len(dis_res)
		result.append(str(total_vm_num))
		for vm in pred_res.keys():
			line = vm + ' ' + str(pred_res[vm])
			result.append(line)
		result.append('')
		result.append(str(total_server_num))
		for i in range(len(dis_res)):
			vm_list = dis_res[i]['vm_list']
			line = ""
			for k in vm_list.keys():
				line += ' ' + k + ' ' + str(vm_list[k])
			line = str(i + 1) + line
			result.append(line)
		return result

	@classmethod
	def get_true_y(cls, fpath, vm_list):
		with open(fpath, 'r') as f:
			lines = f.read().split('\n')
		machines = filter(lambda x: len(x) > 0 and x.split('\t')[1].strip() in vm_list, lines)
		tmp = map(lambda x: x.split('\t')[1].strip(), machines)
		res = Counter(tmp)
		for vm in vm_list:
			if vm not in res.keys():
				res.setdefault(vm, 0)
		return res

	@classmethod
	def _data_preprocess(cls, data):
		mean = sum(data) / len(data)
		return data


### 自己写一个小型numpy
class Numpy(object):

	class nlist(list):
		def __init__(self, lis=None):
			list.__init__([])
			if lis is not None:
				for x in lis: self.append(x)

		def __div__(self, num): return map(lambda x: 1.0 * x / num, self)	

		def __mul__(self, num): return map(lambda x: x * num, self)

		def __add__(self, lis): return map(add, self, lis)


	def __init__(self, lis=None):
		if lis is None:
			self._data = None
			self.row = 0; self.col = 0
		else:
			self._data = lis
			if isinstance(lis[0], list):
				self.row = len(lis)
				self.col = len(lis[0])
			else:
				self.row = 1
				self.col = len(lis)

	def __iter__(self):
		return self

	def __mul__(self, x):
		ret = 0.0
		NumberTypes = (types.IntType, types.LongType, types.FloatType)
		if isinstance(x, NumberTypes):
			for i in xrange(len(self._data)):
				if isinstance(self._data[i], list):
					for j in xrange(len(self._data[i])):
						if isinstance(self._data[i][j], list):
							for k in xrange(len(self._data[i][j])):
								self._data[i][j][k] *= x
						else:
							self._data[i][j] *= x
				else:
					self._data[i] *= x
			ret = Numpy(self._data)
		else:
			ret_row = self.row; ret_col = x.col
			if self.col != x.row:
				raise Exception("Array dimension %s, %s must be equal!" % (self.col , x.row))
			n = len(self._data)
			lis = [[0] * ret_col for i in range(ret_row)]
			for i in xrange(0, ret_row):
				for j in xrange(0, ret_col):
					for k in xrange(0, self.col):
						if not isinstance(self._data[i], list):
							lis[i][j] += self._data[i] * x.to_list()[k][j]
						elif not isinstance(x.to_list()[k], list):
							lis[i][j] += self._data[i][k] * x.to_list()[k]
						else:
							lis[i][j] += self._data[i][k] * x.to_list()[k][j]
			ret = Numpy(lis)
		return ret

	def __div__(self, x):
		NumberTypes = (types.IntType, types.LongType, types.FloatType)
		if isinstance(x, NumberTypes):
			for i in xrange(len(self._data)):
				self._data[i] /= x
		return self

	def __getitem__(self, index): return self._data[index]

	def _list_mul(self, l1, l2): return sum(map(mul, l1, l2))

	def to_list(self): return self._data

	def get_col(self, x, idx):
		ret = []
		lis = x.to_list()
		for i in range(len(lis)):
			for j in range(len(lis[i])):
				if j != idx:
					continue
				ret.append(lis[i][j])
		return ret

	def ones(self, x):
		if isinstance(x, tuple):
			ret =  [[1.0 for i in range(x[1])] for i in range(x[0])]
		else:
			ret =  [1.0 for i in range(x)]
		return Numpy(ret)

	def zeros(self, x):
		if isinstance(x, tuple):
			ret =  [[0.0 for i in range(x[1])] for i in range(x[0])]
		else:
			ret =  [0.0 for i in range(x)]
		return Numpy(ret)

	def sum(self, x):
		return sum(x)

	def random(self, x):
		if isinstance(x, tuple):
			ret = [[random.normalvariate(0, 1) for i in range(x[1])] for i in range(x[0])]
		else:
			ret =  [random.normalvariate(0, 1) for i in range(x)]
		return Numpy(ret)

	def dot(self, x):
		pass

	def outer(self, x):
		pass




if __name__=='__main__':
	x = Numpy(None).ones((2, 2)) 
	y = Numpy(None).ones((2, 2)) * (1.0 / 2)
	z = x * y 
	print z.to_list()

		


