# coding=utf-8
#
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

from helper.args_parser import stock_codes
from checkpoints import CHECKPOINTS_DIR

# 这个文件是拿来测试代码用的, 估计这个script文件夹都是拿来测试代码用的
def load_profits(market='stock'):
    with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'NaiveLSTM', market, 'model_label.json')) as fp:
        _label = np.array(json.load(fp))

    with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'NaiveLSTM', market, 'model_y.json')) as fp:
        _y_naive_lstm = np.array(json.load(fp))

    # with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'DualAttnRNN', market, 'model_y.json')) as fp:
    #     _y_dual_attn_rnn = np.array(json.load(fp))
    #
    # with open(os.path.join(CHECKPOINTS_DIR, 'SL', 'TreNet', market, 'model_y.json')) as fp:
    #     _y_tre_net = np.array(json.load(fp))
    return _label, _y_naive_lstm


label, y_naive_lstm   = load_profits('stock')


row, col = int(math.ceil(len(stock_codes) / 2)), int(math.ceil(len(stock_codes) / 2))
plt.figure(figsize=(20, 15))
for index, code in enumerate(stock_codes):
    plt.subplot(row * 100 + col * 10 + (index + 1))
    plt.title(code)
    plt.plot(label[:, index], label="Real")
    # plt.plot(y_tre_net[:, index], label="TreNet")
    plt.plot(y_naive_lstm[:, index], label="JIMCD")
    # plt.plot(y_dual_attn_rnn[:, index], label="DualAttnRNN")
    plt.legend(loc='upper left')
plt.show()


