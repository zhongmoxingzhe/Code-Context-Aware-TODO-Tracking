import os
import numpy as np
import pandas as pd
import pickle
import time
import datetime
import random
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertConfig
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import collections
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) /len(labels_flat)


def cal_precision(preds, dir_path, keyword='todo'):
    diff_valid, msg_valid, flag_valid = load_data(dir_path, 'test', keyword)
    if len(preds) < len(flag_valid):
        flag_valid = flag_valid[:len(preds)]
    print("Precision: {0:.3f}".format(precision_score(np.array(flag_valid), preds, pos_label=1)))

def cal_f1(preds, dir_path, keyword='todo'):
    diff_valid, msg_valid, flag_valid = load_data(dir_path, 'test', keyword)
    if len(preds) < len(flag_valid):
        flag_valid = flag_valid[:len(preds)]
    f1 = f1_score(np.array(flag_valid), preds, average='binary')  # 正确传入真实标签和预测
    file_path = "./result_f1.txt"
    with open(file_path, "a", encoding="utf-8") as file:
        file.write("f1: {0:.3f}\n".format(f1))
    print("f1: {0:.3f}".format(f1))


def cal_recall(preds, dir_path, keyword='todo'):
    diff_valid, msg_valid, flag_valid = load_data(dir_path, 'test', keyword)
    if len(preds) < len(flag_valid):
        flag_valid = flag_valid[:len(preds)]
    print("Recall: {0:.3f}".format(recall_score(np.array(flag_valid), preds, pos_label=1)))


def cal_auc(preds_prob, dir_path, keyword='todo'):
    diff_valid, msg_valid, flag_valid = load_data(dir_path, 'test', keyword)
    if len(preds_prob) < len(flag_valid):
        flag_valid = flag_valid[:len(preds_prob)]
    file_path = "./result_auc.txt"
    auc = roc_auc_score(np.array(flag_valid), preds_prob[:, 1])
    with open(file_path,"a",encoding="utf-8") as file:
        file.write("AUC: {0:.3f}\n".format(auc))
    print("AUC: {0:.3f}".format(auc))


def cal_ce(preds_prob, dir_path, keyword='todo'):
    diff_valid, msg_valid, flag_valid = load_data(dir_path, 'test', keyword)
    if len(preds_prob) < len(flag_valid):
        flag_valid = flag_valid[:len(preds_prob)]
    tmp_probs = np.asarray(preds_prob[:, 1] - preds_prob[:, 0])
    ind_preds = np.argsort(-tmp_probs)
    get_num = int(0.2 * len(flag_valid))
    get_ind = ind_preds[:get_num]
    get_preds_probs = preds_prob[get_ind]
    get_test_flag = np.array(flag_valid)[get_ind]
    get_preds = np.argmax(get_preds_probs, 1)
    assert len(get_preds) == len(get_test_flag)
    # print("Cost-effectiveness: {0:.3f}".format(recall_score(get_test_flag, get_preds, pos_label=1)))
    count_pos = 0
    total_pos = flag_valid.count(1)
    for p, t in zip(get_preds, get_test_flag):
        # if the approach not give a positive lable
        if t == 1:
            count_pos += 1
    file_path = "./cost_efficiency.txt"
    with open(file_path,"a",encoding="utf-8") as file:
        file.write("cost efficiency: {0:.3f}\n".format(float(count_pos / total_pos)))
    print("Cost-effectiveness: {0:.3f}".format(float(count_pos / total_pos)))


def read_file(path):
    """load lines from a file"""
    sents = []
    with open(path, 'r') as f:
        for line in f:
            sents.append(str(line.strip()))
    return sents


def load_data(dir_path='../dataset/', set_type='train', keyword='todo'):
    # train, valid, test
    if "java" in dir_path:
        language = "java"
    elif "python" in dir_path:
        language = "python"
    diff_file = read_file(dir_path + set_type + "_" + keyword + "_" + language + ".diff")
    msg_file = read_file(dir_path + set_type + "_" + keyword + "_" + language + ".msg")
    flag_file = read_file(dir_path + set_type + "_" + keyword + "_" + language + ".flag")
    flag_file = [int(i) for i in flag_file]
    return diff_file, msg_file, flag_file

def load_path(dir_path='../dataset/', set_type='train', keyword='todo'):
    # train, valid, test
    if "java" in dir_path:
        language = "java"
    elif "python" in dir_path:
        language = "python"
    diff_file = dir_path + set_type + "_" + keyword + "_" + language + ".diff"
    msg_file = dir_path + set_type + "_" + keyword + "_" + language + ".msg"
    flag_file = dir_path + set_type + "_" + keyword + "_" + language + ".flag"
    print("loading...:",flag_file)
    return diff_file, msg_file, flag_file


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second
    elapsed_rounded = int(round(elapsed))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    JAVA_TGT_DIR = "./top_repos_10000/new_java/"
    PYTHON_TGT_DIR = "./top_repos_10000/new_python/"
    dir_path = PYTHON_TGT_DIR
    diff_file, msg_file, flag_file = load_data(dir_path, "train", "todo")
