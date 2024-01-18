import sys
import os
import time
import socket

# originally import
import torch
import matplotlib.pyplot as plt
import numpy as np

data_path = os.getcwd() + '/eps/mpnn/base/origin/mae/'
if not os.path.exists(data_path):
    os.makedirs(data_path)
'''
a = np.load(data_path + 'MPNN_B3LYP_6-31pgs_small.npy')
b = np.load(data_path + 'MPNN_B3LYP_6-31pgs_middle.npy')
c = np.load(data_path + 'MPNN_B3LYP_6-31pgs_large.npy')
'''
a = np.load(data_path + 'MPNN_B3LYP_6-31g.npy')
b = np.load(data_path + 'MPNN_B3LYP_6-31gs.npy')
c = np.load(data_path + 'MPNN_B3LYP_6-31pgs.npy')

pic_name = data_path + '/' + "MPNN_B3LYP_origin_MAE" + '.eps'# + "_mol_size"
# title = "MPNN_" + "B3LYP" #+ "_" + mol_size
x = np.arange(0, 350)
# x1 = np.arange(0, 250)
# plt.title(title)
plt.xlabel("Epoch numbers",fontsize=15)
plt.ylabel("MAE",fontsize=15)
# plt.ylim((0, 1))

plt.plot(x,a,label='6-31G')
plt.plot(x,b,label='6-31G*')
plt.plot(x,c,label='6-31+G*')
'''
plt.plot(x,a,label='small')
plt.plot(x,b,label='middle')
plt.plot(x,c,label='large')
'''
plt.legend()
plt.grid()
plt.savefig(pic_name)
plt.show()