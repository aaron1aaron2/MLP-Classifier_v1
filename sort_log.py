import os
import re
import json
import pandas as pd

log_folder = 'output/20210428'
pattern = 'Nm(\d+)_E(\d.\d+)_B(\d+)_Dr(\d.\d)_Nor(.+)$'

file = 'Nm300_E0.1_B100_Dr0.8_NorMin-Max'
# re.match(pattern, file)

result = pd.DataFrame()
for file in os.listdir(log_folder):
    BEST = [i for i in  os.listdir(os.path.join(log_folder, file)) if i.find('BEST')!=-1][0]
    config = json.load(open(os.path.join(log_folder, file, 'configures.json')))
    error_dt = json.load(open(os.path.join(log_folder, file, 'error_record.json')))
    report_dt = json.load(open(os.path.join(log_folder, file, 'report.json')))
    timeuse = json.load(open(os.path.join(log_folder, file, 'time_record.json')))['total_time']

    df = pd.DataFrame(columns=config.keys(), data=[config.values()])

    epoch = int(re.search('epoch-(\d+)_', BEST)[1])
    df['BEST_epoch'] = epoch

    if re.search('loss-(\d.\d+)_', BEST) !=None:
        loss = re.search('loss-(\d.\d+)_', BEST)[1]
        df['BEST_loss'] = loss
    else:
        df['BEST_loss'] = ''

    if re.search('loss-(\d.\d+)_', BEST) !=None:
        accuracy = re.search('accuracy-(\d.\d+)', BEST)[1]
        df['BEST_accuracy'] = accuracy
    else:
        df['BEST_accuracy'] = ''


    df['BEST_train_error'] = error_dt['train_error_y'][epoch]
    df['BEST_test_error'] = error_dt['test_error_y'][epoch]

    epoch = str(epoch)
    df['BEST_train_accuracy'] = report_dt[epoch]['train']['accuracy']
    df['BEST_test_accuracy'] = report_dt[epoch]['test']['accuracy']

    df['BEST_train_macro_avg_precision'] = report_dt[epoch]['train']['macro avg']['precision']
    df['BEST_test_macro_avg_precision'] = report_dt[epoch]['test']['macro avg']['precision']
    df['BEST_train_macro_avg_recall'] = report_dt[epoch]['train']['macro avg']['recall']
    df['BEST_test_macro_avg_recall'] = report_dt[epoch]['test']['macro avg']['recall']
    df['BEST_train_macro_avg_f1-score'] = report_dt[epoch]['train']['macro avg']['f1-score']
    df['BEST_test_macro_avg_f1-score'] = report_dt[epoch]['test']['macro avg']['f1-score']

    df['timeuse'] = timeuse
    result = result.append(df)

if not os.path.exists('output/result'):
    os.makedirs('output/result')

result.to_excel('output/result/20200428.xlsx')