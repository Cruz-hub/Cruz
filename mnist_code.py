import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

import os
# 在代码开头添加（或保存模型前）
os.makedirs('model', exist_ok=True)  # 自动创建目录，已存在则跳过


#####数据加载#####

train_data=dataset.MNIST(
    root='mnist' ,                                # 数据集下载/保存的根目录（当前目录下的mnist文件夹）
    train=True  ,                                 # 加载训练集（若为False则加载测试集）
    transform=transforms.ToTensor(),              # 将PIL图像或numpy数组转换为张量（Tensor）
    download=True
)

test_data=dataset.MNIST(
    root='mnist' ,                                
    train=False  ,                                # 加载训练集（若为False则加载测试集）
    transform=transforms.ToTensor(),             
)

#print(train_data)
#print(test_data)

#########分批加载#######
train_data=data_utils.DataLoader(dataset=train_data,   #DataLoader加载数据
                                 batch_size=64,
                                 shuffle=True,)
test_data=data_utils.DataLoader(dataset=test_data,
                                 batch_size=64,
                                 shuffle=False,)

#print(train_data)
#print(test_data)

cnn=CNN()


#######损失函数#########
loss_func=torch.nn.CrossEntropyLoss()

######优化函数###########
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.001)

######训练过程##############
#epoc：一次训练数据全部训练一遍
for epoch in range(10):  
    for index,(images,labels) in enumerate(train_data):   #索引、图像数据和标签数据
    #print(index)                                      # enumerate会为每个批次添加一个索引 index，从 0 开始
    #print(images)                                     #打印当前批次的图像数据
    #print(labels)
    #images=images()
    #labels=labels()
   #前向传播
       outputs=cnn(images)
   #传入输出层节点和真实标签来计算损失函数
       loss=loss_func(outputs,labels)
   #清空梯度
       optimizer.zero_grad()
   #反向传播
       loss.backward()
       optimizer.step()
       print('当前为第{}轮，当前批次为{}/{},loss为{}'.format(epoch+1,index+1,len(train_data),loss.item()))
    
############测试集验证#########3
    loss_test=0
    rightValue=0
    for index2,(images,labels) in enumerate(test_data):
        images=images
        labels=labels
        outputs=cnn(images)
        #print(outputs)
        #print(outputs.size())
        #print(labels)
        #print(labels.size())

        loss_test += loss_func(outputs,labels)

        _,pred = outputs.max(1)
        print(pred) 
        rightValue += (pred==labels).sum().item()
        # eq(): 把两个张量中每一个元素进行对比，如果相等，对应位置为True；否则为False，返回一个张量
        print((pred==labels).sum().item())

        print('当前为第{}轮测试集验证，当前批次为{}/{},loss为{}，准确率是{}'.format(epoch+1,index2+1,len(test_data),loss_test,rightValue/len(test_data)))
       
torch.save(cnn.state_dict(), 'model/mnist_model.pkl')




                                             #退出循环，只处理第一个批次的数据
   
