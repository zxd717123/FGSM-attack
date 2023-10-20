import torch
import numpy as np
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data
from torch.autograd import Function
import gc
from scipy import sparse
import os
import random
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

global MAXMIN_V, MAXMIN_PQg, BRANFT, BUS_SLACK  # 平衡节点
global Ybus, baseMVA  # node的300*300矩阵，baseMVA用来控制范围在1pc(perunit)
global Real_Ptrain, Real_Qtrain, Real_Vmtrain, Real_Vatrain, Real_PQdtrain
global scale_vm, VmUb, VmLb, bus_slack, bus_PQg  # 幅值，幅值max,幅值min,平衡节点,节点的有功和无功发电
global flagVm, flagVa, DELTA, flag_hisv, Nbus, Ntest

## whether there is GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Let's use", torch.cuda.device_count(), "GPUs!")

## load system name
REPEAT = 1  # for speedup test: number of repeated computation
s_epoch = 800  # minimum epoch for .pth model saving
p_epoch = 10  # print loss for each "p_epoch" epoch
model_version = 1  # version of model
Nbus = 300  # number of buses(node) 节点上有load和generator或者有可能啥也没有
sys_R = 2  # test case name
flag_test = 0  # 0-train model; 1-test well-trained model
flag_hisv = 0  # 1-use historical V to calculate dV;0-use predicted V to calculate dV
EpochPg = 6  # maximum epoch of Vm 幅值
EpochQg = 6  # maximum epoch of Va 相位角
batch_size_training = 10  # mini-batch size for training
batch_size_test = 1  # mini-batch size for test

## hyperparameters
Lrm = 1e-4  # learning rate for Vm 幅值的学习率
Lra = 1e-4  # learning rate for Va
k_dV = 1  # coefficient for dVa & dVm in post-processing
DELTA = 1e-4  # threshold of violation

# hidden layers for voltage magnitude (Vm) prediction
if Nbus == 300:  # 300的节点，Vm，Va有四层隐藏层
    khidden_Pg = np.array([8, 4, 2], dtype=int)
    khidden_Qg = np.array([8, 4, 2], dtype=int)
    Neach = 12000
else:
    khidden_Pg = np.array([8, 4, 2], dtype=int)
    khidden_Qg = np.array([8, 4, 2], dtype=int)
    Neach = 8000

Ntrain = int(4 * Neach)  # 0-48000 80%
Nsample = int(5 * Neach)  # 48000-600000 20%
Ntest = int(Neach)  # 12000


# name of hidden layers for result saving
nmLPg = 'LPg'
for i in range(khidden_Pg.shape[0]):
    nmLPg = nmLPg + str(khidden_Qg[i])
    print(nmLPg)  # Lm8,Lm86..Lm8642 每个隐藏层都有名字

LPg = khidden_Pg.shape[0]  # number of hidden layers
print(LPg)  # VM 4层隐藏层 (总线为300时)

# hidden layers for voltage angles (Va) prediction
nmLQg = 'LQg'
for i in range(khidden_Qg.shape[0]):
    nmLQg = nmLQg + str(khidden_Qg[i])

LQg = khidden_Qg.shape[0]  # number of hidden layers
print(LQg)  # VA 4层隐藏层 (总线为300时)

# results name
PATHPg = './modelpg' + str(Nbus) + 'r' + str(sys_R) + 'N' + str(model_version) + nmLPg + 'E' + str(EpochPg) + '.pth'
PATHQg = './modelqg' + str(Nbus) + 'r' + str(sys_R) + 'N' + str(model_version) + nmLQg + 'E' + str(EpochQg) + '.pth'
PATHPgs = './modelpg' + str(Nbus) + 'r' + str(sys_R) + 'N' + str(model_version) + nmLPg
PATHQgs = './modelqg' + str(Nbus) + 'r' + str(sys_R) + 'N' + str(model_version) + nmLQg
resultnm = './res_' + str(Nbus) + 'r' + str(sys_R) + 'M' + str(model_version) + 'H' + str(flag_hisv) + 'NT' + str(
    Ntrain) \
           + 'B' + str(batch_size_training) + 'Em' + str(EpochPg) + 'Ea' + str(EpochQg) + nmLPg + nmLQg + 'rp' + str(
    REPEAT) + '.mat'

# load data case 300 large variance
mat = scipy.io.loadmat('./data/XY_case300real.mat')
load_idx = np.squeeze(mat['load_idx']).astype(int) - 1
print(load_idx)  # 哪些节点有负载 1235789...
matpara = scipy.io.loadmat('./data/pglib_opf_case' + str(Nbus) + '_ieeer' + str(sys_R) + '_para.mat')

# power system parameters
RPd0 = mat['RPd']  # Pd active load
RQd0 = mat['RQd']  # Qd reactive load
RPg = mat['RPg']  # Pg active generation
RQg = mat['RQg']  # Qg reactive generation
Ybus = matpara['Ybus']  # Ybus可以用下面这俩表示
Yf = matpara['Yf']  # from更省空间 剔除值为零的节点
Yt = matpara['Yt']  # to更省空间 剔除零的节点
bus = matpara['bus']
gen = matpara['gen']
gencost = matpara['gencost']  # generation cost
branch = matpara['branch']  # 节点i到节点j的路径
baseMVA = matpara['baseMVA']  # 用于scale
bus_slack = np.where(bus[:, 1] == 3)
bus_slack = np.squeeze(bus_slack)
BUS_SLACK = torch.from_numpy(bus_slack).long()
print('bus_slack', bus_slack)


# input data: only contain non-zeros loads 非零负载
# 返回一个一维数组,包含RPd0第一行中（abs）绝对值不为零的元素的索引(where)。
# squeeze调用只是确保我们得到一个一维数组
idx_Pd = np.squeeze(np.where(np.abs(RPd0[0,:]) > 0), axis=0)
idx_Qd = np.squeeze(np.where(np.abs(RQd0[0,:]) > 0), axis=0)
print(idx_Qd)
print(idx_Pd)
# 在RPd0矩阵的所有行,但是只取出idx_Pd中索引对应的列
print(RPd0[:, idx_Pd])
print(RPd0[:, idx_Pd].shape) # 300个中有199个非零active load
bus_Pd = np.squeeze(load_idx[idx_Pd]).astype(int) - 1
bus_Qd = np.squeeze(load_idx[idx_Qd]).astype(int) - 1
# input data 将active load数组和reactive load数组连接（级联）
# axis = 1 内括号级联 (60000, 374) 在第二维拼接（非零有功负载和无功负载）所以是374
x = np.concatenate((RPd0[:, idx_Pd], RQd0[:, idx_Qd]), axis = 1)/baseMVA
print(x)
print('x', x.shape)

# 去掉Pg所有0的列（69变成57）
# idx_Pg = np.squeeze(np.where(np.abs(RPg[0, :]) > 0), axis=0)
# RPg = RPg[:, idx_Pg]

# # RQg进行归一化
# RQgUb = np.amax(RQg)
# RQgLb = np.amin(RQg)
# RQg = (RQg - RQgLb) / (RQgUb - RQgLb)  # scaled Qg
#
# RPgUb = np.amax(RPg)
# RPgLb = np.amin(RPg)
# RPg = (RPg - RPgLb) / (RPgUb - RPgLb)  # scaled Pg

# 创建索引数组-洗牌
indices = np.arange(len(x))

# 洗牌操作
np.random.shuffle(indices)

# 使用洗牌后的索引获取洗牌后的数据
shuffled_x = x[indices]
shuffled_yPg = RPg[indices]
shuffled_yQg = RQg[indices]
Nbus_Pg = RPg.shape[1]
Nbus_Qg = RQg.shape[1]

# Pg Qg data: only contain generators that are turned on
idxPg = np.squeeze(np.where(gen[:, 3] > 0), axis=0)
print(idxPg)  # 只包含启动的 active generation的索引
idxQg = np.squeeze(np.where(gen[:, 1] > 0), axis=0)
print(idxQg)  # 只包含启动的 reactive generation的索引

# Pd Qd of samples 采样
print(Nsample)  # 60 300
RPd = np.zeros((Nsample,Nbus))
RQd = np.zeros((Nsample,Nbus))
# RPd是一个60行300列的矩阵,load_idx指定取出第i列和第..列，一共201列
# RPd0 (60000,201) RPd(60,300)
RPd[:, load_idx] = RPd0[0:Nsample]
RQd[:, load_idx] = RQd0[0:Nsample]
print(RPd)
print(RQd.shape)

del RPd0, RQd0
gc.collect()

yPg_tensor = torch.from_numpy(shuffled_yPg).float()
yQg_tensor = torch.from_numpy(shuffled_yQg).float()

# 计算 yPg_tensor 的均值和标准差
yPg_tensor_mean = torch.mean(yPg_tensor)
yPg_tensor_std = torch.std(yPg_tensor)

# 对 yPg_tensor 进行标准化
yPg_tensor_normalized = (yPg_tensor - yPg_tensor_mean) / yPg_tensor_std
yPg_tensor = yPg_tensor_normalized

# 计算 yQg_tensor 的均值和标准差
yQg_tensor_mean = torch.mean(yQg_tensor)
yQg_tensor_std = torch.std(yQg_tensor)

# 对 yQg_tensor 进行标准化
yQg_tensor_normalized = (yQg_tensor - yQg_tensor_mean) / yQg_tensor_std
yQg_tensor = yQg_tensor_normalized

# loss function
criterion = nn.MSELoss()

# convert training data to tensor
# 非零有功负载和无功负载的级联 （60000，374）
x_tensor = torch.from_numpy(shuffled_x).float()
xtrain = x_tensor[0: Ntrain]
yPgtrain = yPg_tensor[0: Ntrain]
yQgtrain = yQg_tensor[0: Ntrain]


# batch data 训练集 0-48
# 将xtrain和yPgtrain封装成一个数据集(Dataset)
# 这里的Dataset包含了两部分: xtrain: 输入特征 yPgtrain: 标签
# 这样封装成的数据集可直接用于PyTorch的训练、验证和测试。
batch_size_training = 32
training_dataset_Pg = Data.TensorDataset(xtrain, yPgtrain)
training_loader_Pg = Data.DataLoader(
        dataset=training_dataset_Pg,
        batch_size=batch_size_training,
        shuffle=False,
    )

training_dataset_Qg = Data.TensorDataset(xtrain, yQgtrain)
training_loader_Qg = Data.DataLoader(
        dataset=training_dataset_Qg,
        batch_size=batch_size_training,
        shuffle=False,
    )

# test data 测试集 48-60
xtest = x_tensor[Ntrain: Nsample]
yPgtest = yPg_tensor[Ntrain: Nsample]
yQgtest = yQg_tensor[Ntrain: Nsample]

# batch data
batch_size_test = 1
# 将xtest和yPgtest封装成一个数据集(Dataset)
# 这里的Dataset包含了两部分: xtest: 输入特征(12000,374) yPgtest: 标签 (12000,300)
# 这样封装成的数据集可直接用于PyTorch的训练、验证和测试。
test_dataset_Pg = Data.TensorDataset(xtest, yPgtest)
test_loader_Pg = Data.DataLoader(
        dataset=test_dataset_Pg,
        batch_size=batch_size_test,
        shuffle=False,
    )

test_dataset_Qg = Data.TensorDataset(xtest, yQgtest)
test_loader_Qg = Data.DataLoader(
        dataset=test_dataset_Qg,
        batch_size=batch_size_test,
        shuffle=False,
    )

print('yPgtrain', torch.min(yPgtrain), torch.max(yPgtrain))
print('yQgtrain', torch.min(yQgtrain), torch.max(yQgtrain))


## NN function
class NetPg(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units, khidden):
        super(NetPg, self).__init__()

        self.num_layer = khidden.shape[0]
        self.hidden_layers = nn.ModuleList()

        for i in range(self.num_layer):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_channels, khidden[i] * hidden_units))
            else:
                self.hidden_layers.append(nn.Linear(khidden[i-1] * hidden_units, khidden[i] * hidden_units))

        self.fc = nn.Linear(khidden[self.num_layer - 1] * hidden_units, output_channels)
        self.fcpredict = nn.Linear(output_channels, output_channels)

        self.activation = nn.ReLU()  # 将激活函数设置为tanh

    def forward(self, x):
        for i in range(self.num_layer):
            x = self.activation(self.hidden_layers[i](x))

        x = self.activation(self.fc(x))
        x_PredPg = self.fcpredict(x)

        return x_PredPg

class NetQg(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units, khidden):
        super(NetQg, self).__init__()

        self.num_layer = khidden.shape[0]
        self.hidden_layers = nn.ModuleList()

        for i in range(self.num_layer):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_channels, khidden[i] * hidden_units))
            else:
                self.hidden_layers.append(nn.Linear(khidden[i-1] * hidden_units, khidden[i] * hidden_units))

        self.fc = nn.Linear(khidden[self.num_layer - 1] * hidden_units, output_channels)
        self.fcperdict = nn.Linear(output_channels, output_channels)

        self.activation = nn.ReLU()  # 将激活函数设置为tanh

    def forward(self, x):
        for i in range(self.num_layer):
            x = self.activation(self.hidden_layers[i](x))

        x = self.activation(self.fc(x))
        x_PredQg = self.fcperdict(x)

        return x_PredQg


# neural setting
# xtrain非零有功和无功负载 (48000,374)
input_channels = xtrain.shape[1]
output_channels_Pg = yPgtrain.shape[1]
output_channels_Qg = yQgtrain.shape[1]

# determine size of hidden layers
if x.shape[1] >= 100:
     hidden_units = 64
elif x.shape[1] > 30:
     hidden_units = 64
else:
     hidden_units = 16

# train model if it is not test
# 0-train model
if flag_test == 0:
    model_Pg = NetPg(input_channels, output_channels_Pg, hidden_units, khidden_Pg)
    optimizer_Pg = torch.optim.Adam(model_Pg.parameters(), lr=Lrm)

    # 选用CPU或者GPU
    if torch.cuda.is_available():
        model_Pg.to(device)
        print('model_Pg.to(device)')

    print('*' * 5 + 'Pg training' + '*' * 5)
    # Training process: Voltage magnitude 幅值
    start_time = time.process_time()
    # EpochVm 6
    train_loss_all = []
    for epoch in tqdm(range(EpochPg)):
        running_loss = 0.0
        step_count = 0
        print_frequency = 100
        losses = []  # 用于存储损失值
        train_loss = 0
        train_num = 0
        # 遍历训练数据加载器(training_loader_vm),逐批次获得训练数据
        # 对训练数据加载器中的每一批数据:step表示batch的索引，train_x表示当前batch的训练特征数据，train_y表示当前batch的训练标签数据
        for step, (train_x, train_y) in enumerate(training_loader_Pg):
            # feedforward 将训练数据拷贝到指定设备上(CPU或GPU),进行加速
            train_x, train_y = train_x.to(device), train_y.to(device)
            # 通过模型进行前向传播,得到预测输出。
            yPgtrain_hat = model_Pg(train_x)

            # if epoch less than specified number/no penalty of V: only MSEloss
            loss = criterion(train_y, yPgtrain_hat)
            # 获取 loss 张量的数值
            running_loss += loss.item()

            # backproprogate
            # 梯度置零,因为反向传播中梯度会累加,所以每个batch需要置0
            optimizer_Pg.zero_grad()
            # 反向传播,计算损失相对于每个参数的梯度
            loss.backward()
            # 根据梯度更新参数,这一步实际上才更新了模型的参数
            optimizer_Pg.step()
            step_count += 1
            # train_x.size(0) -> batch size
            # 每个样本的损失值乘以批次大小(10)，从而得到了整个批次的总损失
            train_loss += loss.item() * train_x.size(0)
            train_num += train_x.size(0)
            running_loss = 0.0
        train_loss_all.append(train_loss / train_num)

        # 每经过 p_epoch（10）轮打印一次当前轮次、累计损失值以及预测值 yvmtrain_hat 的最小值和最大值
        if (epoch + 1) % p_epoch == 0:
            print('epoch', epoch + 1, running_loss, torch.min(yPgtrain_hat).detach(), torch.max(yPgtrain_hat).detach())

        # save trianed model
        # 每过 100 轮（从 s_epoch 开始），保存一次训练好的模型
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            torch.save(model_Pg.state_dict(), PATHPgs + 'E' + str(epoch + 1) + 'F' + str(flagVm) + '.pth',
                       _use_new_zipfile_serialization=False)

    # save trianed model
    torch.save(model_Pg.state_dict(), PATHPg, _use_new_zipfile_serialization=False)

    # 可视化损失函数的变换情况
    # plt.figure(figsize=(8, 6))
    # plt.plot(train_loss_all, 'ro-', label='Train loss')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.show()


if flag_test == 0:
    # Training process: voltage angle
    model_Qg = NetQg(input_channels, output_channels_Qg, hidden_units, khidden_Qg)
    optimizer_Qg = torch.optim.Adam(model_Qg.parameters(), lr=Lra)
    model_Qg.to(device)
    print('model_Qg.to(device)')
    print('*' * 5 + 'Qg training' + '*' * 5)
    # EpochVa 2
    train_loss_all = []
    for epoch in tqdm(range(EpochQg)):
        running_loss = 0.0
        step_count = 0
        print_frequency = 100
        losses = []  # 用于存储损失值
        train_loss = 0
        train_num = 0
        for step, (train_x, train_y) in enumerate(training_loader_Qg):
            # feedforward
            train_x, train_y = train_x.to(device), train_y.to(device)
            yQgtrain_hat = model_Qg(train_x)
            loss = criterion(train_y, yQgtrain_hat)
            running_loss = running_loss + loss.item()

            # backproprogate
            optimizer_Qg.zero_grad()
            loss.backward()
            optimizer_Qg.step()
            step_count += 1
            # train_x.size(0) -> batch size
            # 每个样本的损失值乘以批次大小(10)，从而得到了整个批次的总损失
            train_loss += loss.item() * train_x.size(0)
            train_num += train_x.size(0)
            running_loss = 0.0
        train_loss_all.append(train_loss / train_num)


        if (epoch + 1) % p_epoch == 0:
            print('epoch', epoch + 1, running_loss, torch.min(yQgtrain_hat).detach(), torch.max(yQgtrain_hat).detach())

        # save trianed model
        if (epoch + 1) % 100 == 0 and (epoch + 1) >= s_epoch:
            torch.save(model_Qg.state_dict(), PATHQgs + 'E' + str(epoch + 1) + 'F' + str(flagVa) + '.pth',
                       _use_new_zipfile_serialization=False)


    # save trianed model
    torch.save(model_Qg.state_dict(), PATHQg, _use_new_zipfile_serialization=False)
    # 可视化损失函数的变换情况
    # plt.figure(figsize=(8, 6))
    # plt.plot(train_loss_all, 'ro-', label='Train loss')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.show()


xtest = xtest.to(device)

# load trained model if it is testing
# 1-test well-trained model
if flag_test == 1:
    # load trained model 幅值
    print('load trained model: model_Pg_load')
    # 定义两个网络模型结构 model_vm 和 model_va,与训练时的模型结构相同
    model_Pg = NetPg(input_channels,output_channels_Pg,hidden_units,khidden_Pg)
    # 加载保存好的模型参数
    model_Pg.load_state_dict(torch.load(PATHPg, map_location=device))
    # 将模型设置为评估模式:model_vm.eval()
    model_Pg.eval()
    # 将模型拷贝到指定设备上
    model_Pg.to(device)
    # load trained model 相位角
    print('load trained model: model_Qg_load')
    model_Qg = NetQg(input_channels,output_channels_Qg,hidden_units,khidden_Qg)
    model_Qg.load_state_dict(torch.load(PATHQg, map_location=device))
    model_Qg.eval()
    model_Qg.to(device)

print('*' * 5 + 'begin repeated calcualtion for time testing' + '*' * 5)
for k in range(REPEAT):
    if (k + 1) % 10 == 0:
        print('REPEAT', k + 1)

    # Predicted data
    time_PredVm_NN = 0
    # yvmtest（12，300）
    yPgtest_hat = torch.zeros((Ntest, Nbus_Pg))
    yPgtest_hats = []
    # 使用enumerate可以同时得到每个batch的索引step和batch的数据(test_x, test_y)
    with torch.no_grad():
      for step, (test_x, test_y) in enumerate(test_loader_Pg):
        test_x = test_x.to(device)
        yPgtest_hat[step] = model_Pg(test_x)
        # 计算模型预测一批数据的时间time_PredVm_NN
        # 计算相对误差

        # error = get_mae(test_y, yvmtest_hat_clip)
        # print("Step {}: Relative Error: {}".format(step, error))

      yPgtest_hat = yPgtest_hat.cpu()
      yPgtest_hats = yPgtest_hat.detach() * yPg_tensor_std + yPg_tensor_mean
    # 对预测结果进行裁剪,限制在历史数据范围内


    # yvatest_hat（12，299）Nbus300
    yQgtest_hat = torch.zeros((Ntest, Nbus_Qg))
    # yvatest_hat = torch.zeros((Ntest, Nbus))
    with torch.no_grad():
      for step, (test_x, test_y) in enumerate(test_loader_Qg):
        test_x = test_x.to(device)
        yQgtest_hat[step] = model_Qg(test_x)

      yQgtest_hat = yQgtest_hat.cpu()
      yQgtest_hats = yQgtest_hat.detach() * yQg_tensor_std + yQg_tensor_mean
    # 对预测结果进行裁剪,限制在历史数据范围内
    # yvatest_hat_clip = get_clamp(yvatest_hats, hisVa_min, hisVa_max)


print('***************DNN Model Summary', '***************')
print('training setting: Ntrain', Ntrain)
print('testing setting:  Ntest', Ntest)
print('EpochVmVa:', EpochPg, 'batch_size:', batch_size_training)
print('learning rate:', '(1)Lrm', Lrm, ' (2)Lra', Lra)
print('NN layer:',  ' khidden_Pg', khidden_Pg * hidden_units, 'khidden_Qg', khidden_Qg * hidden_units)
print(model_Pg)
print(model_Qg)

print('*' * 5 + 'end repeated calcualtion for time testing' + '*' * 5)

# performance evaluation
# no revison
# (12,300)

# mae_Vmtest = get_mae(yvmtests, yvmtest_hat_clip.detach())
# mre_Vmtest_clip = get_rerr(yvmtests,yvmtest_hat_clip.detach())
yPgtests = yPgtest * yPg_tensor_std + yPg_tensor_mean
#yPgtests = yPgtests.numpy()
#yPgtest_hats = yPgtest_hats.detach().numpy()

print('*' * 5 + 'To evaluate the performance of a predictive mode_Pg' + '*' * 5)
error_Pg = torch.abs(yPgtests - yPgtest_hats)
# 取第一行并计算绝对误差
MAE_Pg = torch.mean(error_Pg)
# 求误差的平均值
print('Pg_MAE：', MAE_Pg)

# mask = yPgtests != 0
# MRE_Pg = np.zeros_like(error_Pg)
# MRE_Pg[mask] = np.abs(error_Pg[mask] / yPgtests[mask])
# # MRE_Pg = np.abs(error_Pg / yPgtest)
# median_relative_error = np.median(MRE_Pg)
# print('Pg_MdAPE:', median_relative_error)


yQgtests = yQgtest * yQg_tensor_std + yQg_tensor_mean
# yQgtests = yQgtests.numpy()
# yQgtest_hats = yQgtest_hats.detach().numpy()

print('*' * 5 + 'To evaluate the performance of a predictive mode_Qg' + '*' * 5)
error_Qg = torch.abs(yQgtests - yQgtest_hats)
# 计算绝对误差
MAE_Qg = torch.mean(error_Qg)
# 求误差的平均值
print('Qg_MAE：', MAE_Qg)

# error_Qg = np.abs(yQgtests - yQgtest_hats)
# relative_error = np.abs(error_Qg / yQgtests)
# # 使用NumPy的median函数计算相对误差的中位数，即MdAPE
# median_relative_error = np.median(relative_error)
# print('Qg_MdAPE:', median_relative_error)


print('*' * 5 + 'Results of testing' + '*' * 5)
#print('Vm_MRE：', get_rerr3(yvmtests, yvmtest_hat_clip))
#print('Va_MRE：',  mre_va_total)
print('Ori_yQgtest', yQgtests[0][0])
# (2000,299)
print('Pre_yQgtest', yQgtest_hats[0][0])
print('Ori_yPgtest', yPgtests[0][0])
print('Pre_yPgtest', yPgtest_hats[0][0])

import matplotlib.pyplot as plt

# 假设 yvatests 是 (2000, 299) 的二维数组，predictions 是相同维度的预测值数组
# yQgtest = yQgtest[0:50]
# yQgtest_hat = yQgtest_hat[0:50]
# index = np.argsort(yQgtest.flatten())
# print(yQgtest.flatten()[index])
# print(yQgtest_hat.flatten()[index])
#
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(yQgtest.size), yQgtest.flatten()[index], 'r', label='Original Y_Qg')
# plt.scatter(np.arange(yQgtest.size), yQgtest_hat.flatten()[index], s=2, c='b', label='Prediction')
# plt.legend(loc='upper left')
# plt.grid()
# plt.xlabel('Index')
# plt.ylabel('Y')
# plt.show()
#
#
# yPgtest = yPgtest[0:50]
# yPgtest_hat = yPgtest_hat[0:50]
# index = np.argsort(yPgtest.flatten())
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(yPgtest.size), yPgtest.flatten()[index], 'r', label='Original Y_Pg')
# plt.scatter(np.arange(yPgtest.size), yPgtest_hat.flatten()[index], s=2, c='b', label='Prediction')
# plt.legend(loc='upper left')
# plt.grid()
# plt.xlabel('Index')
# plt.ylabel('Y')
# plt.show()

# print(yPgtest)
# print(yPgtest_hat)
# print(yQgtest)
# print(yQgtest_hat)

def fgsm_attack(model, loss, test_x, test_y, eps):
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_x.requires_grad = True

    outputs = model(test_x)

    model.zero_grad()
    cost = loss(outputs, test_y).to(device)
    cost.backward()

    test_x_adv = test_x + eps * test_x.grad.detach().sign()
    #test_x_adv = torch.clamp(test_x_adv, 0, 1)

    return test_x_adv


print('*' * 5 + 'Use FGSM for testing' + '*' * 5)
for k in range(REPEAT):
    if (k + 1) % 10 == 0:
        print('REPEAT', k + 1)

    # Predicted data
    # yvmtest（12，300）
    yPgtest_hat_adv = torch.zeros((Ntest, Nbus_Pg))
    # 使用enumerate可以同时得到每个batch的索引step和batch的数据(test_x, test_y)
    for step, (test_x, test_y) in enumerate(test_loader_Pg):
        loss = nn.MSELoss()
        eps = 0.008
        test_x = fgsm_attack(model_Pg, loss, test_x, test_y, eps).to(device)
        test_y = test_y.to(device)
        yPgtest_hat_adv[step] = model_Pg(test_x)

    yPgtest_hat_adv = yPgtest_hat_adv.cpu()
    yPgtest_hats_adv = yPgtest_hat_adv.detach() * yPg_tensor_std + yPg_tensor_mean

    # yvatest_hat（12，299）Nbus300
    yQgtest_hat_adv = torch.zeros((Ntest, Nbus_Qg))
    for step, (test_x, test_y) in enumerate(test_loader_Qg):
        loss = nn.MSELoss()
        eps = 0.008
        test_x = fgsm_attack(model_Qg, loss, test_x, test_y, eps).to(device)
        test_y = test_y.to(device)
        yQgtest_hat_adv[step] = model_Qg(test_x)

    yQgtest_hat_adv = yQgtest_hat_adv.cpu()
    yQgtest_hats_adv = yQgtest_hat_adv.detach() * yQg_tensor_std + yQg_tensor_mean

print('*' * 5 + 'performance evaluation After FSGM_attack ' + '*' * 5)
#yPgtest_hat_adv = yPgtest_hat_adv.detach().numpy()
#yPgtests = torch.from_numpy(yPgtests).float()
# error_Pg_adv = np.abs(yPgtest - yPgtest_hat_adv)
# # 取第一行并计算绝对误差
# MAE_Pg_adv = np.mean(error_Pg_adv)
# MAE_Pg_adv = np.round(MAE_Pg_adv, 4)
# # 求误差的平均值
# print('Pg_MAE：', MAE_Pg_adv)

error_Pg_adv = torch.abs(yPgtests - yPgtest_hats_adv)
# 取第一行并计算绝对误差
MAE_Pg_adv = torch.mean(error_Pg_adv)
# 求误差的平均值
print('Pg_MAE：', MAE_Pg_adv)

#yQgtests = torch.from_numpy(yQgtests).float()
#yQgtest_hats_adv = yQgtest_hats_adv.numpy()
error_Qg_adv = torch.abs(yQgtests - yQgtest_hats_adv)
# 计算绝对误差
MAE_Qg_adv = torch.mean(error_Qg_adv)
# 求误差的平均值
print('Qg_MAE：', MAE_Qg_adv)

print('\n','*'*20,'Testing: ','*'*20)
print('  mae_Pgtest', MAE_Pg.detach().numpy(), 'mae_Qgtest', MAE_Qg.detach().numpy())
print('\n','*'*20,'Testing after attack: ','*'*20)
print('  mae_Pgtest', MAE_Pg_adv.detach().numpy(), 'mae_Qgtest', MAE_Qg_adv.detach().numpy())


print('\n','*'*20,'Scatter plot: ','*'*20)
# 第一个图表
yPgtests = yPgtests[0:30]
yPgtests = yPgtests[(yPgtests > 0) & (yPgtests < 20)]
yPgtests = yPgtests.detach().numpy()
yPgtest_hats_adv = yPgtest_hats_adv[0:30]
yPgtest_hats_adv = yPgtest_hats_adv.detach().numpy()
yPgtest_hats = yPgtest_hats[0:30]
yPgtest_hats = yPgtest_hats.detach().numpy()
index = np.argsort(yPgtests.flatten())
# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# 在第一个子图上绘制曲线
ax1.set_ylim(0, 20)  # 设置纵坐标范围
ax1.set_yticks(np.arange(0, 21, 1))  # 设置纵坐标刻度间隔
ax1.plot(np.arange(yPgtests.size), yPgtests.flatten()[index], 'r', label='Original Y_Pg')
ax1.scatter(np.arange(yPgtests.size), yPgtest_hats_adv.flatten()[index], s=2, c='b', label='Prediction')
ax1.legend(loc='upper left')
ax1.grid()
ax1.set_xlabel('Index')
ax1.set_ylabel('Y_Pg')

# 在第二个子图上绘制曲线
ax2.set_ylim(0, 20)  # 设置纵坐标范围
ax2.set_yticks(np.arange(0, 21, 1))  # 设置纵坐标刻度间隔
ax2.plot(np.arange(yPgtests.size), yPgtests.flatten()[index], 'r', label='Original Y_Pg')
ax2.scatter(np.arange(yPgtests.size), yPgtest_hats.flatten()[index], s=2, c='b', label='Prediction')
ax2.legend(loc='upper left')
ax2.grid()
ax2.set_xlabel('Index')
ax2.set_ylabel('Y_Pg')

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4)

# 显示图表
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 第一个图表
yQgtests = yQgtests[0:30]
yQgtests = yQgtests[(yQgtests > 0) & (yQgtests < 20)]
yQgtests = yQgtests.detach().numpy()
# 小范围测试
yQgtest_hats = yQgtest_hats[0:30]
yQgtest_hats = yQgtest_hats.detach().numpy()
yQgtest_hats_adv = yQgtest_hats_adv[0:30]
yQgtest_hats_adv = yQgtest_hats_adv.detach().numpy()
index = np.argsort(yQgtests.flatten())

# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# 在第一个子图上绘制曲线
ax1.set_ylim(0, 20)  # 设置纵坐标范围
ax1.set_yticks(np.arange(0, 21, 1))  # 设置纵坐标刻度间隔
ax1.plot(np.arange(yQgtests.size), yQgtests.flatten()[index], 'r', label='Original Y_Qg')
ax1.scatter(np.arange(yQgtests.size), yQgtest_hats_adv.flatten()[index], s=2, c='b', label='Prediction')
ax1.legend(loc='upper left')
ax1.grid()
ax1.set_xlabel('Index')
ax1.set_ylabel('Y_Qg')

# # 在第二个子图上绘制曲线
ax2.set_ylim(0, 20)  # 设置纵坐标范围
ax2.set_yticks(np.arange(0, 21, 1))  # 设置纵坐标刻度间隔
ax2.plot(np.arange(yQgtests.size), yQgtests.flatten()[index], 'r', label='Original Y_Qg')
ax2.scatter(np.arange(yQgtests.size), yQgtest_hats.flatten()[index], s=2, c='b', label='Prediction')
ax2.legend(loc='upper left')
ax2.grid()
ax2.set_xlabel('Index')
ax2.set_ylabel('Y_Qg')

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4)

# 显示图表
plt.show()

