import sys
import os
import time
import socket

# originally import
import torch
import matplotlib.pyplot as plt
import numpy as np

data_path = os.getcwd() + '/eps/lstm/base/improve1/mae'
'''
a = np.load(data_path + '/' + 'LSTM_B3LYP_6-31pgs_small.npy')
b = np.load(data_path + '/' + 'LSTM_B3LYP_6-31pgs_middle.npy')
c = np.load(data_path + '/' + 'LSTM_B3LYP_6-31pgs_large.npy')
'''
# d = np.load(data_path + 'LSTM_P38_631gss2.npy')

a = np.load(data_path + '/' + 'LSTM_B3LYP_6-31g.npy')
b = np.load(data_path + '/' + 'LSTM_B3LYP_6-31gs.npy')
c = np.load(data_path + '/' + 'LSTM_B3LYP_6-31pgs.npy')


# pic_name = data_path + '/' + "P38_631gss2" + '.eps'# + "_mol_size"
pic_name = data_path + '/' + "LSTM_B3LYP_bg_MAE" + '.eps'# + "_mol_size"_size
# title = "MPNN_" + "B3LYP" #+ "_" + mol_size
x = np.arange(0, 350)
# x1 = np.arange(0, 350)
# plt.title(title)
plt.xlabel("Epoch numbers",fontsize=15)
plt.ylabel("MAE",fontsize=15)
# plt.ylim((0, 1))


plt.plot(x,a,label='6-31G')
plt.plot(x,b,label='6-31G*')
plt.plot(x,c,label='6-31+G*')

# plt.plot(x,d,label='6-31G**')
'''
plt.plot(x,a,label='small')
plt.plot(x,b,label='middle')
plt.plot(x,c,label='large')
'''
plt.legend()
plt.grid()
plt.savefig(pic_name)
plt.show()