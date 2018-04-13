# coding=utf-8
import copy
import conf
from models import LogistModel, NNModel, HMM
from data_helper import DataHelper
from distribution import Distributor

def predict_vm(ecs_lines, input_lines):
    '''
    预测虚拟机数量主函数
    @input:  ecs_lines:训练数据中虚拟机数量的时间分布
            input_lines:物理机的限制及需要预测的虚拟机规格、最优化目标（CPU或者内存）与时间
    @return: result: 预测结果：每台物理机放置的虚拟机数
    '''
    result = []
    if ecs_lines is None:
        print 'ecs information is none'
        return result
    if input_lines is None:
        print 'input file information is none'
        return result

    mod = 'submmit'   ## train or submmit
    save_model = False
    true_y = None
    train_data, _ = DataHelper.split_data(ecs_lines)
    total_train_x, total_train_y = DataHelper.trans_data_format(train_data)
    need_predict_vm = DataHelper.get_need_predict(input_lines)

    if mod == 'train':
        save_model = True
        true_y = DataHelper.get_true_y(conf.test_data_path, need_predict_vm)
    pred_res = dict()
    for vm in need_predict_vm:
        t_train_x, t_train_y = DataHelper.get_specific_vm(total_train_x, total_train_y, vm)
        train_x, train_y, test_x, test_y = DataHelper.sliding_window(t_train_x, t_train_y, vm)
        print vm, train_x, train_y, test_x, test_y
        feat_dim = len(train_x[0])
        if feat_dim < 6:
            Model = LogistModel
        else:
            Model = HMM(conf.hideen_state, feat_dim, conf.HMM_iter)

        model_path = Model.train(vm, train_x, train_y, save_model=False)   ### 每次提交代码都要将save_model改为true
        
        if Model == NNModel or Model == SimpleNetwork:
            pred = Model.predict(model_path, test_x)
        elif isinstance(Model, HMM):
            pred = Model.decode(model_path, test_x)
        else:
            pred = Model.predict(vm, model_path, total_train_x, use_history_mean=True)   ### use_history_mean采用历史平均数据作为预测值
        print vm, pred, test_y[0]
        pred_res.setdefault(vm, pred)
    if true_y != None:
        Model.eval_on_test_data(pred_res, true_y)
    pred_vm = copy.deepcopy(pred_res)
    dis_result = Distributor.distribute_BFD(pred_res, input_lines)
    result = DataHelper.write_final_res(pred_vm, dis_result)
    return result


### 0.机器学习模型时间序列回归模型：预测下一时间段不同规格虚拟机的数量分布（目前先用逻辑回归做baseline，之后可以考虑用ARIMA)
### 1.最优化问题：背包（装箱）将预测出的虚拟机分配给物理机:使得机器数量最小，资源利用率最大

### 数据划分 6 weeks train, 1 week test
### 1.1-2.19 train   2.20-2.27 test
### 2.20-3.19 train  3.20-3.27 test
### 3.20-4.19 train  4.20-4.27 test
### 4.20-5.19 train  5.20-5.27 test




