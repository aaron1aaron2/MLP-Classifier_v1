# encoding: utf-8
"""
@author: yen-nan ho
@contact: aaron1aaron2@gmail.com
"""
import os
import time
import argparse
import tracemalloc 
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

from utils import load_mnist, log_string, save_json, save_filename
from model import MiddleLayer, OutputLayer, Dropout
from sklearn.metrics import classification_report

class mymodel:
    def __init__(self, wb_width, n_in, n_mid, n_out, eta, class_num, dropout_rate):
        self.class_num = class_num
        self.eta = eta
        self.input_layer = MiddleLayer('input_layer', wb_width, n_in, n_mid)
        self.dropout_1 = Dropout('dropout_1', dropout_rate)
        self.middle_layer = MiddleLayer('middle_layer', wb_width, n_mid, n_mid)
        self.dropout_2 = Dropout('dropout_2', dropout_rate)
        self.output_layer = OutputLayer('output_layer', wb_width, n_mid, n_out)
         
    def forward_propagation(self, x, is_train):
        self.input_layer.forward(x)
        self.dropout_1.forward(self.input_layer.y, is_train)
        self.middle_layer.forward(self.dropout_1.y)
        self.dropout_2.forward(self.middle_layer.y, is_train)
        self.output_layer.forward(self.dropout_2.y)
        self.pred = self.encode_to_class(self.output_layer.y)

        # 如果不是 train 將模型預測結果回傳
        if not is_train:
            return self.pred

    def backpropagation(self, t):
        self.t = self.class_to_encode(t, self.class_num)
        self.output_layer.backward(self.t)
        self.dropout_2.backward(self.output_layer.grad_x)
        self.middle_layer.backward(self.dropout_2.grad_x)
        self.dropout_1.backward(self.middle_layer.grad_x)
        self.input_layer.backward(self.dropout_1.grad_x)

    def update_wb(self):
        self.input_layer.update(self.eta)
        self.middle_layer.update(self.eta)
        self.output_layer.update(self.eta)

    def encode_to_class(self, output):
        '''取出每個位置 one-hot 的最大值的位置作為預測類別'''
        output = np.argmax(output, axis=1) # 
        return output+1 # 1~10

    def class_to_encode(self, t, class_num):
        '''轉成 one-hot 才可以進入模型 backword'''
        t = np.eye(class_num)[t-1]
        return t

    def get_error(self, label, pred):
        '''必須將標記轉換成 0-1 之間，cross entropy 才不會變負數'''
        assert len(label) == len(pred), "預測與目標 shape 大小不同"
        num = len(label)
        Max = label.max()
        Min = label.min()
        label_nor = (label - Min) / (Max - Min)
        pred_nor = (pred - Min) / (Max - Min)

        return -np.sum(label_nor  * np.log(pred_nor+ 1e-7)) / num  # 交叉熵誤差誤差

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default = 'data',
                        help = 'mnist 資料夾')
    parser.add_argument('--output_folder', type=str, default = 'output/test',
                        help = '結果記錄位置')

    parser.add_argument('--n_mid', type=int, default = 100,
                        help = '中間層的神經元數量')
    parser.add_argument('--wb_width', type=int, default = 0.01,
                        help = '初始權重與偏值的範圍')
    parser.add_argument('--eta', type=float, default = 0.01,
                        help = '學習率')
    parser.add_argument('--batch_size', type=int, default = 100,
                        help = 'batch 大小')  
    parser.add_argument('--dropout_rate', type=float, default = 0.5,
                        help = 'dropout 大小') 
    parser.add_argument('--normalisation', type=str, default = 'Min-Max',
                        help = '[Min-Max, z-score, log]') 

    parser.add_argument('--num_epoch', type=int, default = 100,
                        help = '最大 epoch')
    parser.add_argument('--interval', type=int, default = 1,
                        help = '多少 epoch 計算一次誤差')

    args = parser.parse_args()

    return args


def load_data(path, normalisation):
    X_train, y_train = load_mnist(path=path, kind='train')
    X_test, y_test = load_mnist(path=path, kind='t10k')

    if normalisation == 'Min-Max':
        Max = X_train.max()
        Min = X_train.min()

        X_train_nor = (X_train - Min) / (Max - Min)
        X_test_nor = (X_test - Min) / (Max - Min)

    elif normalisation == 'z-score':
        Mean = np.mean(X_train)
        Std = np.std(X_train)  

        X_train_nor = (X_train - Mean) / Std
        X_test_nor = (X_test - Mean) / Std

    elif normalisation == 'log':
        X_train_nor = np.ma.log(X_train)
        X_train_nor = X_train_nor.filled(0) 

        X_test_nor = np.ma.log(X_test)
        X_test_nor = X_test_nor.filled(0) 

    else:
        return X_train, y_train, X_test, y_test

    return X_train_nor, y_train, X_test_nor, y_test

def main():
    # 參數讀取 & 前置作業 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Start_time = time.time()
    args = get_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
    	print(f'target already build at {args.output_folder}')
    	exit() # 如果已經跑過了，跳過

    save_json(args.__dict__, os.path.join(args.output_folder, 'configures.json'), indent=2)
    logger = open(os.path.join(args.output_folder, 'log.txt'), 'w')
    log_string(logger, f'[args]\n{str(args)[10 : -1]}\n')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 讀取檔案 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    tracemalloc.start() # 計算 memory 用
    start_ramuse, _ = tracemalloc.get_traced_memory()  
    X_train, y_train, X_test, y_test = load_data(path=args.data_folder, normalisation=args.normalisation)
    log_string(logger, f'[shape of data]\nX_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}\n')
    end_ramuse, peak = tracemalloc.get_traced_memory()  
    tracemalloc.stop()

    # 傳換目標 1-10
    target_ls = list(set(y_train))
    target_dt = {v:k+1 for k,v in  enumerate(target_ls)} 
    log_string(logger, f'[target]\n{target_ls}\n')

    y_train = np.array(list(map(lambda x: target_dt[x], y_train)))
    y_test = np.array(list(map(lambda x: target_dt[x], y_test)))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 參數準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    n_in = X_train.shape[1]
    n_out = len(target_ls)
    n_mid = args.n_mid
    n_train = len(y_train)
    
    n_batch = n_train // args.batch_size
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 資料儲存空間 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    error_record_dt = {'train_error_x':[], 'train_error_y':[], 'test_error_x':[], 'test_error_y':[]}
    time_record_dt = {'Start_time':Start_time, 'eval_time':[], 'train_time':[]}
    memory_record_dt = {'load_data':{'start':start_ramuse, 'end':end_ramuse, 'peak':peak},
                        'model':{},
                        'train':{}
                        }
    report_dt = {}
    BEST = {'epoch':0, 'loss':float('inf'), 'accuracy':0}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 讀取模型 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(logger, 'compiling model...')

    tracemalloc.start() # 計算 memory 用
    start_ramuse, _ = tracemalloc.get_traced_memory()  
    model = mymodel(wb_width=args.wb_width, n_in=n_in, n_mid=n_mid, n_out=n_out, 
                    eta=args.eta, class_num=n_out, dropout_rate=args.dropout_rate)
    end_ramuse, peak = tracemalloc.get_traced_memory()  
    tracemalloc.stop()

    memory_record_dt['model'] = {'start':start_ramuse, 'end':end_ramuse, 'peak':peak}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # 開始訓練 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    log_string(logger, 'training model...')
    for i in range(num_epoch):        
        # 訓練模型 -------------
        tracemalloc.start()
        train_start_time = time.time()

        # memory checkpoint 1
        memory_record_dt['train'][i] = [tracemalloc.get_traced_memory()]
        index_random = np.arange(n_train)
        np.random.shuffle(index_random)  # 索引洗牌
        for j in range(n_batch):
            # 取出小批次
            mb_index = index_random[j*batch_size : (j+1)*batch_size]
            x = X_train[mb_index, :]
            label = y_train[mb_index]

            # 前向傳播與反向傳播
            model.forward_propagation(x, True)
            model.backpropagation(label)

            # 更新權重與偏值
            model.update_wb()

        # memory checkpoint 2
        memory_record_dt['train'][i].append(tracemalloc.get_traced_memory())

        train_time = round((time.time() - train_start_time)/60, 2)
        time_record_dt['train_time'].append(train_time)

        # 驗證模型 -------------
        eval_start_time = time.time()

        train_pred = model.forward_propagation(X_train, False)
        train_loss = model.get_error(label=y_train, pred=train_pred)
        train_report = classification_report(y_train, model.pred, output_dict=True)

        test_pred = model.forward_propagation(X_test, False)
        test_loss = model.get_error(label=y_test, pred=test_pred)
        test_report = classification_report(y_test, model.pred, output_dict=True)

        # memory checkpoint 3
        memory_record_dt['train'][i].append(tracemalloc.get_traced_memory())

        # 儲存結果 -------------
        report_dt.update({i:{'train':train_report, 'test':test_report}})

        error_record_dt['test_error_x'].append(i)
        error_record_dt['test_error_y'].append(train_loss) 
        error_record_dt['train_error_x'].append(i)
        error_record_dt['train_error_y'].append(test_loss) 

        eval_time = round((time.time()-eval_start_time)/60, 4)
        time_record_dt['eval_time'].append(eval_time)

        total_time = round((time.time() - Start_time)/60, 2)
        time_record_dt['total_time'] = total_time
        per_ep_time = round(total_time/(i+1), 2)
        estimated_time = round(per_ep_time*(num_epoch-i))

        # memory checkpoint 4
        memory_record_dt['train'][i].append(tracemalloc.get_traced_memory())
        tracemalloc.stop()

        # 顯示結果 -------------
        if i%args.interval == 0:
            msg = f"[Epoch]: {i}/{num_epoch}{'='*50}\n" + \
                  f"Train loss: {round(train_loss,3)} | Test loss: {round(test_loss,3)} | train_time:{train_time} min | eval_time: {eval_time} min " + \
                  f"| Train accuracy: {round(train_report['accuracy'],3)} |Test accuracy: {round(test_report['accuracy'],3)}\n" + \
                  f">> total time use: {total_time} min | speed: {per_ep_time}  min/epoch | estimated_time: {estimated_time} min\n"
            log_string(logger, msg)
    
            save_json(error_record_dt, os.path.join(args.output_folder, 'error_record.json'))
            save_json(time_record_dt, os.path.join(args.output_folder, 'time_record.json'))
            save_json(report_dt, os.path.join(args.output_folder, 'report.json'))
            save_json(memory_record_dt, os.path.join(args.output_folder, 'memory_record.json'))
        
        # 如果當前 epoch 的 test loss 比之前最佳的 test loss 好，更新最佳結果
        if BEST['loss'] > test_loss:
            BEST.update({'epoch':i, 'loss':round(test_loss, 3), 'accuracy':round(test_report['accuracy'],3)})
            log_string(logger, f'update BEST Test loss {round(test_loss, 3)} at epoch {i}')
            
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    save_filename('BEST', BEST, args.output_folder)

    

if __name__ == "__main__":
    main()