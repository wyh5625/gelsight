import matplotlib.pyplot as plt # plotting library
import os
import numpy as np

data1 = np.load('./model/loss.npy',allow_pickle=True)
data1= data1.item()

data2 = np.load('./model/loss2.npy',allow_pickle=True)
data2= data2.item()

data3 = np.load('./model/loss_noxy.npy',allow_pickle=True)
data3= data3.item()

fig, axs = plt.subplots(1, 2, figsize=(12, 6)
                        )

axs[0].plot(data1['train_loss'],label='train_loss1')

axs[1].plot(data1['val_loss'],label='val_loss1')

axs[0].plot(data2['train_loss'],label='train_loss2')

axs[1].plot(data2['val_loss'],label='val_loss2')

axs[0].plot(data3['train_loss'],label='train_loss3')

axs[1].plot(data3['val_loss'],label='val_loss3')

plt.legend()
plt.show()