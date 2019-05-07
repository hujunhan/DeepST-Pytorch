#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys

sys.path.append('../')
import deepst.dataloader as dataloader
import deepst.model as model
import numpy as np
import torch
import h5py
from torch.utils import data

# In[2]:


print(torch.cuda.is_available())

# In[3]:


c, p, t, y, d = dataloader.get_all_data()

# In[4]:


c.dtype

# In[5]:


y_max = y.max()
y_min = y.min()
print(y_max, y_min)

# In[6]:


f = dataloader.get_feature_data(d)
print(len(f))


# In[7]:


def minmax(data):
	data = 1. * (data - 0) / (1270 - 0)
	data = data * 2.0 - 1.0
	return data


def rescale(data, max=1270.0, min=0.0):
	data = (data + 1.) / 2.
	data = 1. * data * (max - min) + min
	return data


# In[8]:


from torch.utils import data


class TaxiBJ(data.Dataset):
	def __init__(self):
		pass

	def __getitem__(self, index):
		return minmax(c[index]), minmax(p[index]), minmax(t[index]), f[index], minmax(y[index])

	def __len__(self):
		return len(c)


# In[9]:


dataset = TaxiBJ()

# In[10]:


loader = data.DataLoader(dataset=dataset, batch_size=1000, shuffle=False)

# In[12]:


ct, pt, tt, ft, yt = dataset[1]

# In[13]:


device = torch.device('cpu')


# In[14]:


class ToTensor(object):
	def __call__(self, c, p, t, f, y):
		ct = torch.from_numpy(c).unsqueeze(0)
		pt = torch.from_numpy(p).unsqueeze(0)
		tt = torch.from_numpy(t).unsqueeze(0)
		ft = torch.from_numpy(f).unsqueeze(0)
		yt = torch.from_numpy(y).unsqueeze(0)
		return ct.to(device), pt.to(device), tt.to(device), ft.to(device), yt.to(device)


# In[15]:


trans = ToTensor()

# In[19]:


net = model.ResNet()
net.double()

# In[20]:


# from tensorboardX import SummaryWriter
#
# with SummaryWriter(comment='ST-ResNet')as w:
# 	w.add_graph(net, (ct, pt, tt, ft))

# In[21]:


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# In[22]:


crit = torch.nn.MSELoss()

# In[12]:


import time

thistime = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
txtname = 'extra/'+thistime + '.txt'

# In[23]:


ff = open(txtname, 'w')

# In[24]:


for i in range(2000):
	for step in enumerate(loader):
		ct, pt, tt, ft, yt = dataset[i]
		ct, pt, tt, ft, yt = trans(ct, pt, tt, ft, yt)
		optimizer.zero_grad()
		out = net(ct, pt, tt, ft)
		loss = crit(out, yt)
		loss.backward()
		optimizer.step()
		# ff.write(str(loss.item())+'\n')
		print(loss.item())
ff.close()

# In[16]:


torch.save(net.state_dict(), 'params_6_14_all_scale.pkl')

# In[17]:


Y = []
f = open('train_6_14.txt', 'r')

# In[18]:


for line in f.readlines():
	# print(line.strip())
	Y.append(float(line.strip()))

# In[19]:


import matplotlib.pyplot as plt

x = range(len(Y))

# In[20]:


plt.bar(x, Y)

# In[21]:


plt.show()

# In[22]:


f.close()

