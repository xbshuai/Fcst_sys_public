import torch as th
import torch.nn as nn
import numpy as np
from MPNN import MPNNModel
from torch.utils.data import DataLoader
from GDataSetTRAIN import TencentAlchemyDataset, batcher
import os
import xlsxwriter
from ModelTool import ModelTool

th.manual_seed(2)


#import Magnification

# MPNN class
class MpnnTool(ModelTool):
    
    def __init__ (self,chemspace,config,suits1,suits2,folder_sdf,folder_mod,target):
        '''
        Construction
        chemspace: Cheimcal space
        config:  need one Configs.Config object for initializing
        '''
        ModelTool.__init__(self,chemspace,config,folder_sdf,target)
        self.suits1=suits1
        self.suits2=suits2
        self.folder_mod=folder_mod


    def eval_each(self,modelname,chemspace="B3LYP_6-31g",path='./',write=False):

        data_each=[]

        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]
        
        tra_size=self.config.tra_size

        if not os.path.exists("tmp"):
           os.mkdir("tmp")
        icount=0
        # The used validing suits            
        tmp1="./tmp/valid-tmp"+chemspace+"MPNN"
        with open(tmp1,'w') as ftmp:
           for suit in self.suits2:
              #print(suit)
              with open(suit,'r') as fsuits:
                 for line in fsuits:
                    icount=icount+1  
                    ftmp.write(line)
              print(suit, " : ", icount)

        print("Total molecules in validing suit : ", icount)            

        dataset=TencentAlchemyDataset(mode='valid',rootdir=path,suits=tmp1,chemspace=chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        loader = DataLoader(dataset     = dataset,
                            batch_size  = self.config.batch_size,
                            collate_fn  = batcher(),
                            shuffle     = False,
                            num_workers = 0)

        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return
        # model = th.load(modelname,map_location=th.device('cpu'))
        model = th.load(modelname)
        model.to(self.device)

        #loss_fn = nn.MSELoss()
        #MAE_fn = nn.L1Loss() 
        model.eval()
        bnums   = []
        times   = []
        preds   = []
        # sdflist = []

        with th.no_grad():
            err=0
            errs=[]
            j=0
            mae=0.0
            with open(tmp1,'r') as ftmp:
                ftmplines=ftmp.readlines()
            #print(ftmplines)   

            for idx,batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                #batch.sdf=batch.basisnum.to(self.device)
                res = model(batch.graph,batch.basisnum)
                res=res.to('cpu')
                #mae = MAE_fn(res, batch.label)
                #w_mae += mae.detach().item()
                reslist=res.numpy()
                #print(reslist)
                reslist=res.tolist()
                batch.label=batch.label.to('cpu')
                batch.basisnum=batch.basisnum.to('cpu')
                timelist=batch.label.numpy()
                #print(timelist)
                timelist=timelist.tolist()
                bnumlist=batch.basisnum.numpy().tolist()

                for i in range(len(reslist)):
                    
                    time=timelist[i][0]
                    ares=reslist[i]
                    bnum=bnumlist[i][0]
                       
                    #print(bnum)
                    times.append(time)
                    preds.append(ares)
                    bnums.append(bnum)
                    #sdflist.append(sdf) 
                    err1=(float(ares)-float(time))/float(time)
                    print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',bnum,' real time : ',time,' predicted time: ',ares, 'err', err1)
                    data_each.append([ftmplines[i].split()[4],bnum,time,ares,err1])
                    ae=abs(time-ares)
                    single_err=ae/time
                    err+=single_err
                    mae+=ae
                    errs.append(single_err)
                    j+=1
            err_mean=err/j
            mae/=j
            errs=np.array(errs)
            variance=errs.var()

            print("MRE：%.4f, MAE: %.4F, VAR: %.4f"%(err_mean,mae,variance))

        if write is True:
            wbname='results_'+self.chemspace+'/'+'tes_mpnn_'+str(tra_size)+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})#测试结果
            worksheet=workbook.add_worksheet("")
            worksheet.write_row('A1',['nbasis','res','real'])
            for i in range(len(bnums)):
                content=[bnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")

        return data_each


    def eval(self,modelname,chemspace="B3LYP_6-31g",path='./',write=False):
        '''
        testing or evaluating 
        path: path of testing suits; default as'data/tes_'+self.chemspace+'.txt'
        write: check whether into xlsx format
        '''
        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]
        
        tra_size=self.config.tra_size

        if not os.path.exists("tmp"):
           os.mkdir("tmp")
        icount=0
        # The used validing suits            
        tmp1="./tmp/valid-tmp"+chemspace+"MPNN"
        with open(tmp1,'w') as ftmp:
           for suit in self.suits2:
              #print(suit)
              with open(suit,'r') as fsuits:
                 for line in fsuits:
                    icount=icount+1  
                    ftmp.write(line)
              print(suit, " : ", icount)

        print("Total molecules in validing suit : ", icount)            

        dataset=TencentAlchemyDataset(mode='valid',rootdir=path,suits=tmp1,chemspace=chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        loader = DataLoader(dataset     = dataset,
                            batch_size  = self.config.batch_size,
                            collate_fn  = batcher(),
                            shuffle     = False,
                            num_workers = 0)

        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return
        # model = th.load(modelname,map_location=th.device('cpu'))
        model = th.load(modelname)
        model.to(self.device)

        #loss_fn = nn.MSELoss()
        #MAE_fn = nn.L1Loss() 
        model.eval()
        bnums   = []
        bnums_s = []
        times   = []
        preds   = []
        with th.no_grad():
            err=0
            errs=[]
            j=0
            mae=0.0
            with open(tmp1,'r') as ftmp:
                ftmplines=ftmp.readlines()

            for idx,batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                batch.basisnums = batch.basisnums.to(self.device)
                res = model(batch.graph,batch.basisnums)
                res=res.to('cpu')
                #mae = MAE_fn(res, batch.label)
                #w_mae += mae.detach().item()
                reslist=res.numpy()
                #print(reslist)
                reslist=res.tolist()
                batch.label=batch.label.to('cpu')
                batch.basisnum=batch.basisnum.to('cpu')
                batch.basisnums = batch.basisnums.to('cpu')
                timelist=batch.label.numpy()
                #print(timelist)
                timelist=timelist.tolist()
                bnumlist=batch.basisnum.numpy().tolist()
                bnumslist = batch.basisnums.numpy().tolist()

                for i in range(len(reslist)):
                    
                    time=timelist[i][0]
                    ares=reslist[i]
                    bnum=bnumlist[i][0]
                    bnum_s = bnumslist[i][0]
                       
                    #print(bnum)
                    times.append(time)
                    preds.append(ares)
                    bnums.append(bnum)
                    bnums_s.append(bnum_s)
                    err1=(float(time)-float(ares))/float(time)
                    print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',bnum,' basis sum num: ',bnum_s,' real time : ',time,' predicted time: ',ares, 'err', err1)
                    ae=abs(time-ares)
                    single_err=ae/time
                    err+=single_err
                    mae+=ae
                    errs.append(single_err)
                    j+=1
            err_mean=err/j
            mae/=j
            errs=np.array(errs)
            variance=errs.var()

            print("MRE：%.4f, MAE: %.4F, VAR: %.4f"%(err_mean,mae,variance))

        if write is True:
            wbname='results_'+self.chemspace+'/'+'tes_mpnn_'+str(tra_size)+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})
            worksheet=workbook.add_worksheet("")
            worksheet.write_row('A1',['nbasis','res','real'])
            for i in range(len(bnums)):
                content=[bnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")

        return [err_mean,mae,variance]    

    def train(self,path='./',mol_size="small"):

        tra_size=self.config.tra_size

        if not os.path.exists("tmp"):
           os.mkdir("tmp")
        icount = icount_s = icount_m = icount_l = 0
        # The used training suits


        tmp1="./tmp/train-tmp_s"
        tmp2="./tmp/train-tmp_m"
        tmp3="./tmp/train-tmp_l"
        
        with open(tmp1,'w') as ftmp_s:
            with open(tmp2,'w') as ftmp_m:
                with open(tmp3,'w') as ftmp_l:
                    for suit in self.suits1:
                        with open(suit,'r') as fsuits:
                            for line in fsuits:
                                temp=line.strip(os.linesep).split()
                                if float(temp[0]) < 200.0 :
                                    icount_s = icount_s + 1
                                    ftmp_s.write(line)
                                elif float(temp[0]) > 400.0 :
                                    icount_l = icount_l + 1
                                    ftmp_l.write(line)
                                else:
                                    icount_m = icount_m + 1
                                    ftmp_m.write(line)
                        
        print("Molecules in small training suit : ", icount_s)
        print("Molecules in middle training suit : ", icount_m)
        print("Molecules in large training suit : ", icount_l)
        print("Total molecules in training suit : ", icount_s + icount_m + icount_l)            
        if mol_size == "small":
            dataset=TencentAlchemyDataset(mode='train',rootdir=path,suits=tmp1,chemspace=self.chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        elif mol_size == "middle":
            dataset=TencentAlchemyDataset(mode='train',rootdir=path,suits=tmp2,chemspace=self.chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        else:
            dataset=TencentAlchemyDataset(mode='train',rootdir=path,suits=tmp3,chemspace=self.chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        '''
        tmp1="./tmp/train-tmp"
        with open(tmp1,'w') as ftmp:
           for suit in self.suits1:
              #print(suit)
              with open(suit,'r') as fsuits:
                 for line in fsuits:
                    icount=icount+1  
                    ftmp.write(line)
              print(suit, " : ", icount)
        print("Total molecules in training suit : ", icount)

        dataset=TencentAlchemyDataset(mode='train',rootdir=path,suits=tmp1,chemspace=self.chemspace,folder_sdf=self.sdf_dir,tra_size=tra_size, target = self.target)
        '''
        loader=DataLoader(dataset     = dataset,
                          batch_size  = self.config.batch_size,
                          collate_fn  = batcher(),
                          shuffle     = False,
                          #drop_last   = True,
                          num_workers = 0)
        model=MPNNModel(device=self.device)
        model.to(self.device)
        model.train()

        loss_fn = nn.L1Loss()
        #MAE_fn = nn.L1Loss()

        optimizer = th.optim.Adam(model.parameters(), lr=self.config.lr)

        targetName=''
        if self.target==2 or self.target==3:
            targetName='tot'
        elif self.target==6:
            targetName='ave'

        modelName     = self.folder_mod + '/' + 'mpnn_' + self.chemspace + '_' + targetName + '.pkl'    
        modelName_tmp = self.folder_mod + '/' + 'mpnn_' + self.chemspace + '_' + targetName + '_tmp.pkl'

        minMre=100.0
        bestEpoch=0
        save_step=10
        y = []
        y1 = []
        for epoch in range(1,self.config.tra_num_epochs+1):
            w_loss = 0
            err    = 0
            errs   = []
            j      = 0
            mae = 0.0
            for idx, batch in enumerate(loader):
                #import pdb
                #pdb.set_trace()
                batch.graph    = batch.graph.to(self.device)
                batch.label    = batch.label.to(self.device)
                batch.basisnum = batch.basisnum.to(self.device)
                batch.basisnums = batch.basisnums.to(self.device)
                res            = model(batch.graph,batch.basisnums)
                
                loss           = loss_fn(res, batch.label.squeeze(-1))
                #mae = MAE_fn(res, batch.label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                res            = res.to('cpu')
                batch.label    = batch.label.to('cpu')

                reslist        = res.detach().numpy()
                reslist        = reslist.tolist()
                #print(reslist)
                
                timelist       = batch.label.detach().numpy()
                timelist       = timelist.tolist()# timelist=[[time1],[time2],...]
                #print(timelist)
                
                for i in range(len(reslist)):
                    ares       = reslist[i]
                    time       = timelist[i][0]
                    mae_single = abs(ares - time)
                    mae += mae_single
                    single_err = abs(ares - time) / time
                    err += single_err
                    errs.append(single_err)
                    j+=1

                #w_mae += mae.detach().item()
                w_loss += loss.detach().item()
            #w_mae /= idx + 1
            mae = mae / j
            y1.append(mae)
            err_mean=err/j
            errs=np.array(errs)
            variance=errs.var()
            y.append(err_mean)

            print("Epoch {:2d}, loss: {:.7f}, mre: {:.7f},variance: {:.4f}, mae:{:.4f}".format(epoch, w_loss / j,
                                                                                               err_mean, variance, mae))

            if epoch%save_step==0:
                th.save(model,modelName_tmp)
                eval_res=self.eval(modelname=modelName_tmp,chemspace=self.chemspace,path=path)
                if eval_res[0]<minMre:
                    th.save(model,modelName)
                    minMre=eval_res[0]
                    bestEpoch=epoch

        print("training done! Best epoch is "+str(bestEpoch))
        print("training done : keep the best model and delete the intermediate models")
        os.remove(modelName_tmp)
        data_path = os.getcwd() + '/eps/mpnn/size/improve2/mre/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path1 = os.getcwd() + '/eps/mpnn/size/improve2/mae/'
        if not os.path.exists(data_path1):
            os.makedirs(data_path1)
        '''
        if self.chemspace == "B3LYP_6-31g":
            np.save(data_path1 + 'MPNN_B3LYP_6-31g', y1)
        elif self.chemspace == "B3LYP_6-31gs":
            np.save(data_path1 + 'MPNN_B3LYP_6-31gs', y1)
        else:
            np.save(data_path1 + 'MPNN_B3LYP_6-31pgs', y1)
        '''
        if mol_size == "small":
            np.save(data_path1 + 'MPNN_B3LYP_6-31pgs_small', y1)
        elif mol_size == "middle":
            np.save(data_path1 + 'MPNN_B3LYP_6-31pgs_middle', y1)
        else:
            np.save(data_path1 + 'MPNN_B3LYP_6-31pgs_large', y1)

        return minMre


    def finetune(self,path='./',linear_dim1=13,linear_dim2=10,epochs=35,lr=0.001):
        '''
        对模型做微调
        path: data目录的上级目录，默认为当前目录,data目录下存放数据集文件
        linear_dim1: 倒数第二个全连接层的输入维度
        linear_dim2: 倒数第一个全连接层的输入维度
        epochs: 训练的轮数
        lr: 学习率
        '''
        print("start")
        tra_size=self.config.tra_size
        dataset = TencentAlchemyDataset(mode='train',rootdir=path,chemspace=self.chemspace,tra_size=tra_size)
        loader = DataLoader(dataset=dataset,
                                    batch_size=self.config.batch_size,
                                    collate_fn=batcher(),
                                    shuffle=False,
                                    num_workers=0)
        if tra_size<4000:
            modeldir='model_'+str(tra_size)+'/'
        else:
            modeldir='model/'
        modelname=modeldir+self.chemspace+'_mpnn.pkl'
        model=th.load(modelname)
        for param in model.parameters():
            param.requires_grad=False
        model.fclayer1=nn.Linear(linear_dim1,linear_dim2)
        model.fclayer2=nn.Linear(linear_dim2,1)
        loss_fn = nn.MSELoss()
        #MAE_fn = nn.L1Loss()
        optimizer = th.optim.Adam(model.parameters(), lr=lr)
        model=model.to(self.device)
        model.train()
        for epoch in range(epochs):

            w_loss= 0
            
            err=0
            errs=[]
            j=0
            for idx, batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                res = model(batch.graph,batch.basisnum)  
                loss = loss_fn(res, batch.label)
                #mae = MAE_fn(res, batch.label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                res=res.to('cpu')
                batch.label=batch.label.to('cpu')
                reslist=res.detach().numpy()
                reslist=reslist.tolist()
                timelist=batch.label.detach().numpy()
                timelist=timelist.tolist()
                
                for i in range(len(reslist)):
                    ares=reslist[i]
                    time=timelist[i][0]
                    single_err=abs(ares-time)/time
                    err+=single_err
                    errs.append(single_err)
                    j+=1

                #w_mae += mae.detach().item()
                w_loss += loss.detach().item()
            #w_mae /= idx + 1
            err_mean=err/j
            errs=np.array(errs)
            variance=errs.var()
            print("Epoch {:2d}, loss: {:.7f},  mre: {:.7f},variance: {:.4f}".format(epoch, w_loss, err_mean,variance))
        th.save(model,modeldir+self.chemspace+'_mpnn_f.pkl')


    def predict(self,root='./',pdata=None,batch_size=32): 
        '''
        调用模型进行推断
        root: data目录的上级目录，默认为当前目录,data目录下存放数据集文件
        pdata: 待预测数据,pdata形式为[[sdf1,sdf2,...],[basisnum1,basisnum2,...]]
        batch_size: batch大小
        '''
        #root代表待测分子的sdf文件所在目录的上级存放目录,pdata为待测数据
        #pdata=[[sdf1,sdf2,...],[basisnum1,basisnum2,...]]
        if pdata is None:
            print("input data is None!")
            return 
        modelname='model/'+self.chemspace+'_mpnn.pkl'
        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return
        model=th.load(modelname,map_location=th.device('cpu')).to(self.device)
        model.eval()
        preds=[]

        dataset=TencentAlchemyDataset(mode='pred',rootdir=root,chemspace=self.chemspace,pdata=pdata)
        loader=DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            collate_fn=batcher(),
                            shuffle=False,
                            num_workers=0)
        model.set_mean_std(dataset.mean,dataset.std,self.device)
        model.to(self.device)
        with th.no_grad():
            for idx,batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label=batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                res=model(batch.graph,batch.basisnum)
                res=res.to('cpu')
                reslist=res.numpy().tolist()
                preds.extend(reslist)
        
        
        return preds