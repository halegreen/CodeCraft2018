# coding=utf-8

class Distributor(object):

	_vm_flavor_info = { 'flavor1' : (1, 1024),
						'flavor2' : (1, 2048),
						'flavor3' : (1, 4096),
						'flavor4' : (2, 2048),
						'flavor5' : (2, 4096),
						'flavor6' : (2, 8192),
						'flavor7' : (4, 4096),
						'flavor8' : (4, 8192),
						'flavor9' : (4, 16384),
						'flavor10' : (8, 8192),
						'flavor11' : (8, 16384),
						'flavor12' : (8, 32768),
						'flavor13' : (16, 16384),
						'flavor14' : (16, 32768),
						'flavor15' : (16, 65536)
						}

	@classmethod
	def _get_infos(cls, pred_res, input_lines):
		CPU_num, MEM_size, _ = input_lines[0].split()
		vm_num = input_lines[2]
		i = 3
		vm_flavors = list(); CPU_array = list(); MEM_array = list()
		while i < len(input_lines):
			line = input_lines[i].strip()
			if len(line) == 0 or line.strip() == 'CPU' or line.strip() == 'MEM':
				break
			vm_flavors.append(line.split()[0].strip())
			CPU_array.append(int(line.split()[1].strip()))
			MEM_array.append(int(line.split()[2].strip()))
			i += 1
		need_optimize = input_lines[-4].strip()
		return vm_flavors, CPU_array, MEM_array, need_optimize

	@classmethod
	def distribute_greedy(cls, pred_res, input_lines):
		'''
		@input: pred_res: key: vm_flavor, value: vm_num
		@return: 
		'''
		CPU_num, MEM_size, _ = input_lines[0].split()
		vm_flavors, CPU_array, MEM_array, need_optimize = cls._get_infos(pred_res, input_lines)
		# print vm_flavors, CPU_array, MEM_array, need_optimize
		server = {'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()}
		physic_machines = list()
		physic_machines.append(server)
		for i in range(len(vm_flavors)):
			while pred_res[vm_flavors[i]] != 0:
				# print ''
				# print physic_machines
				cur_server = physic_machines.pop()
				if CPU_array[i] <= cur_server['CPU_left'] and MEM_array[i] <= cur_server['MEM_left']:
					cur_server['CPU_left'] -= CPU_array[i]
					cur_server['MEM_left'] -= MEM_array[i]
					if vm_flavors[i] not in cur_server['vm_list'].keys():
						cur_server['vm_list'][vm_flavors[i]] = 0
					cur_server['vm_list'][vm_flavors[i]] += 1
					physic_machines.append(cur_server)
				else:
					physic_machines.append(cur_server)
					physic_machines.append({'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()})
					n_cur_server = physic_machines.pop()
					n_cur_server['CPU_left'] -= CPU_array[i]
					n_cur_server['MEM_left'] -= MEM_array[i]
					if vm_flavors[i] not in n_cur_server['vm_list'].keys():
						n_cur_server['vm_list'][vm_flavors[i]] = 0
					n_cur_server['vm_list'][vm_flavors[i]] += 1
					physic_machines.append(n_cur_server)
				pred_res[vm_flavors[i]] -= 1
		return physic_machines

	@classmethod
	def distribute_BFD(cls, pred_res, input_lines):  ### 降序最佳适应算法(BFD)
		CPU_num, MEM_size, _ = input_lines[0].split()
		vm_flavors, _, _, need_optimize = cls._get_infos(pred_res, input_lines)
		vm_flavors = sorted(vm_flavors)
		vm_flavors.reverse()
		physic_machines = list()
		physic_machines.append({'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()})

		for i in range(len(vm_flavors)):
			while pred_res[vm_flavors[i]] != 0:
				src_min = int(CPU_num) if need_optimize == 'CPU' else int(MEM_size) * 1024 
				for k in range(len(physic_machines)):   ###遍历所有的箱子，找到最适应的箱子放入 
					cpu_diff = physic_machines[k]['CPU_left'] - cls._vm_flavor_info[vm_flavors[i]][0]
					mem_diff = physic_machines[k]['MEM_left'] - cls._vm_flavor_info[vm_flavors[i]][1]
					if cpu_diff < 0 or mem_diff < 0 or physic_machines[k]['CPU_left'] <= 0 or physic_machines[k]['MEM_left'] <= 0:
						if k == len(physic_machines) - 1:
							physic_machines.append({'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()})
							input_machine_idx = len(physic_machines) - 1
							break
						continue
					if need_optimize == 'CPU' and cpu_diff < src_min:
						input_machine_idx = k
						src_min = cpu_diff
					elif need_optimize == 'MEM' and mem_diff < src_min:
						input_machine_idx = k
						src_min = mem_diff
				if vm_flavors[i] not in physic_machines[input_machine_idx]['vm_list'].keys():
					physic_machines[input_machine_idx]['vm_list'][vm_flavors[i]] = 0
				pred_res[vm_flavors[i]] -= 1
				physic_machines[input_machine_idx]['vm_list'][vm_flavors[i]] += 1
				physic_machines[input_machine_idx]['CPU_left'] -= cls._vm_flavor_info[vm_flavors[i]][0]
				physic_machines[input_machine_idx]['MEM_left'] -= cls._vm_flavor_info[vm_flavors[i]][1]
		return physic_machines

	@classmethod
	def distribute_dp(cls, pred_res, input_lines):   ### 看成背包问题的动态规划，还没写完，有点问题
		CPU_num, MEM_size, _ = input_lines[0].split()
		vm_flavors, CPU_array, MEM_array, need_optimize = cls._get_infos(pred_res, input_lines)
		vm_flavors = sorted(vm_flavors)
		vm_flavors.reverse()
		### 初始化
		physic_machines = list()
		cur_machine = {'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()}
		last_machine = cur_machine
		dp = [[0 for col in range(CPU_num + 1)] for row in range(len(vm_flavors))]
		print dp

		if need_optimize == 'CPU':
			for i in range(0, len(vm_flavors)):
				if last_machine['CPU_left'] == 0 or last_machine['MEM_left'] == 0:
					cur_machine = {'CPU_left' : int(CPU_num), 'MEM_left' : int(MEM_size) * 1024, 'vm_list' : dict()}  
				else:
					cur_machine = last_machine

				for j in range(total_CPU + 1, 0, -1):
					if pred_res[vm_flavors[i]] == 0:
						break

					k_min = min(pred_res[vm_flavors[i]], cur_machine['CPU_left'] / cls._vm_flavor_info[vm_flavors[i]][0])
					k_min = min(t_min, cur_machine['MEM_left'] / cls._vm_flavor_info[vm_flavors[i]][1])

					for k in range(k_min + 1, 0, -1):
						if pred_res[vm_flavors[i]] == 0:
							break

						if i == 0 and j > k * cls._vm_flavor_info[vm_flavors[i]][0]:
							dp[i][j] = k * cls._vm_flavor_info[vm_flavors[i]][0]

						elif j >= k * cls._vm_flavor_info[vm_flavors[i]][0]:
							if dp[i - 1][j - k * cls._vm_flavor_info[vm_flavors[i]][0]] + \
								cls._vm_flavor_info[vm_flavors[i]][0] > dp[i - 1][j]:
								dp[i][j] = dp[i - 1][j - k * cls._vm_flavor_info[vm_flavors[i]][0]] + \
											cls._vm_flavor_info[vm_flavors[i]][0]
								### 分配虚拟机，更新资源
								if vm_flavors[i] not in cur_machine.keys():
									cur_machine['vm_list'].setdefault(vm_flavors[i], k)
								cur_machine['vm_list'][vm_flavors[i]] = k
								cur_machine['CPU_left'] -= cls._vm_flavor_info[vm_flavors[i]][0]
								cur_machine['MEM_left'] -= cls._vm_flavor_info[vm_flavors[i]][1]
								pred_res[vm_flavors[i]] -= k
							else:
								dp[i][j] = dp[i - 1][j]
				physic_machines.append(cur_machine)
				last_machine = cur_machine

		return physic_machines



