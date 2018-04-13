# coding=utf-8

model_param_path = 'data/model_param/'
test_data_path = 'data/TestData_2015.2.20_2015.2.27.txt'

### logist model params
iterations_num = 100
opimize_type = 'stoc_grad_descent'  ##grad_descent, stoc_grad_descent, smooth_stoc_grad_descent
alpha = 0.5

### nueral network model params
NN_nh = 20
NN_no = 1
learning_rate = 0.05
correct = 0.1

simple_network_topology = [10, 3, 1]
simple_network_eta = 0.1
simple_network_alpha = 0.015

hideen_state = 4
HMM_iter = 20
