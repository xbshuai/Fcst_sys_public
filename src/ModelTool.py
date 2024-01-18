import torch as th
import abc
import matplotlib.pyplot as plt
import numpy as np

class ModelTool(metaclass=abc.ABCMeta):

    '''
    模型工具类的抽象基类
    '''

    def __init__(self,chemspace,config,sdf_dir,target):
        '''
        构造方法
        chemspace: 化学空间
        config: 需要一个Configs.Config对象来初始化
        '''
        self.chemspace=chemspace
        self.config=config
        self.sdf_dir=sdf_dir
        #self.mol_name=mol_name
        self.target=target
        '''
        self.device = th.device('cpu')
        '''
        if th.cuda.is_available():
            print("CUDA available")
            self.device=th.device('cuda:0')
        else:
            print("CUDA unavailable")
            self.device=th.device('cpu')



    def gen_learning_curve(self,tra_losses,tra_mres,tra_maes,val_losses,val_mres,val_maes):
        '''
        生成学习曲线图并保存
        tra_losses: 训练集损失数组
        tra_mres: 训练集平均相对误差数组
        tra_maes: 训练集平均绝对误差数组
        val_losses: 验证集损失数组
        val_mres: 验证集平均相对误差数组
        val_maes: 验证集平均绝对误差数组
        '''
        valid_interval=self.config.valid_interval
        tra_losses=np.array(tra_losses)
        tra_mres=np.array(tra_mres)
        tra_maes=np.array(tra_maes)
        val_losses=np.array(val_losses)
        val_mres=np.array(val_mres)
        val_maes=np.array(val_maes)

        font={'family':'Times New Roman',
        'weight':'normal',
        'size': 18}

        epochs=np.arange(1,self.config.tra_num_epochs+1).astype(np.int)
        epochs_val=np.arange(valid_interval,len(val_losses)*valid_interval+1,valid_interval).astype(np.int)
        fig=plt.figure(num=1,figsize=(12,8),dpi=80)
        ax1=fig.add_subplot(2,1,1) # loss-epoch曲线图
        ax2=fig.add_subplot(2,1,2) # 误差-epoch曲线图
        plt.subplots_adjust(hspace=0.5)

        ax1.set_title('(a)',font)
        ax1.set_xlabel('Epochs',font)
        ax1.set_ylabel('MSE Loss',font)
        ax1.scatter(epochs,tra_losses,s=80,c='r',marker='x')
        ax1.plot(epochs,tra_losses,linestyle='-',color='#FFA07A',label='Training Set',linewidth=4)
        ax1.scatter(epochs_val,val_losses,s=80,c='y',marker='+')
        ax1.plot(epochs_val,val_losses,linestyle='-',color='#e3f51e',label='Valid Set',linewidth=4)
        ax1.legend(loc='upper right',prop=font)

        ax2.set_title('(b)',font)
        ax2.set_xlabel('Epochs',font)
        ax2.set_ylabel('MRE',font)
        ax2.scatter(epochs,tra_mres,s=80,c='#FF6347',marker='.')
        ax2.plot(epochs,tra_mres,linestyle='-',color='#FF8C00',label='Training Set',linewidth=4)
        ax2.scatter(epochs_val,val_mres,s=80,c='#00FFFF',marker='D')
        ax2.plot(epochs_val,val_mres,linestyle='-',color='#00BFFF',label='Valid Set',linewidth=4)
        ax2.legend(loc='upper right',prop=font)
        modelname=self.__class__.__name__.split('Tool')[0]
        plt.savefig('plots/'+modelname+'_'+self.chemspace+'.jpg',dpi=400,bbox_inches='tight')
        #plt.show()

#    @abc.abstractmethod
#    def train(self,path):
#        '''
#        训练模型
#        path: 训练集路径，一般无需手动指定
#        '''
#        raise NotImplementedError("method train() has not been implemented!")

    @abc.abstractmethod
    def eval(self,path,write):
        '''
        测试模型
        path: data目录的上级目录，默认为当前目录,data目录下存放数据集文件
        write: 是否将测试结果写入xlsx
        '''
        raise NotImplementedError("method eval() has not been implemented!")

#    @abc.abstractmethod
#    def predict(self,pdata,batch_size):
#        '''
#        调用模型进行推断
#        pdata: 待预测数据，形式为[[sdf1,sdf2,...],[nbasis1,nbasis2,...]],sdf1表示该sdf文件的路径
#        batch_size: batch大小
#        '''
#        raise NotImplementedError("method predict() has not been implemented!")
