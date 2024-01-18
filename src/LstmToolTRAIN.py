from rdkit import Chem
import json
import os
import torch
import torch.nn as nn
import gensim
import BiLSTM
import time as systime
import torch.optim as optim
import numpy as np
import xlsxwriter
from ModelTool import ModelTool
import matplotlib.pyplot as plt

torch.manual_seed(2)

import basis_set_exchange as bse
#import Magnification

# LSTM Class

class LstmTool(ModelTool):

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
        

    @staticmethod
    def readData(paths,sdf_dir,tra_size,target,basis=""): 
        '''
        Get the data, Nbasis - CPU times - smiles - sdf names, from suits
        '''
        
        print(" ==> sdf_dir in readData",sdf_dir)
        print(" ==> paths",paths)
        bnum = []
        bnums = []
        times = []
        slist = []  # smiles list
        names = []
        count = 0
        # interval=4000/tra_size
        for path in paths:
            for line in open(path,'r'):
                count+=1
                if count>tra_size:
                    break
                temp=line.strip(os.linesep).split()
                time=float(temp[target])#时间

                sdf=sdf_dir + "/" + temp[4].split('_')[0]+".sdf"
                for i in range(len(temp)):
                    if temp[i] == 'contracted':
                        basisnum_s = float(temp[i+2])
                        basisnum_p = float(temp[i+3])
                        basisnum_d = float(temp[i+4])
                        basisnum_f = float(temp[i+5])
                        basisnum_g = float(temp[i+6])
                        basisnum_h = float(temp[i+7].strip(']'))
                        break

                basisnum = [basisnum_s, basisnum_p, basisnum_d, basisnum_f, basisnum_g, basisnum_h]#各个轨道总数目
                basisnums = float(temp[0])
                
                #print("sdf ",temp[4].split('_')[0], "basis", basisnum, basisnum2)
                
                bnum.append(basisnum)
                bnums.append(basisnums)
                
                name=temp[4]#.split('_')[1]
                #print("name : ",name)
                names.append(name)
                #basisnums.append(basisnum)
                times.append(time)
                #s=sdf_dir+'/'+name+'.sdf'   # updated in YJMA version
                s=sdf_dir+'/'+name.split('_')[0]+'.sdf'
                #print("modified s : ",s)
                suppl=Chem.SDMolSupplier(s) # 读取单个sdf文件,存成的是一个列表的形式，支持列表的操作。
                for mol in suppl:
                    smiles=Chem.MolToSmiles(mol) # 读取单个smiles字符串
                    slist.append(smiles)
        return bnum,bnums,times,slist,names


    @staticmethod
    def sdf_to_smiles(sdfs): 
        '''
         sdf ==> smiles
        '''
        slist=[]
        for sdf in sdfs:
            suppl=Chem.SDMolSupplier(sdf)
            for mol in suppl:
                smiles=Chem.MolToSmiles(mol)
                slist.append(smiles)
        return slist


    @staticmethod
    def seg(slist):
        '''
         smiles ==> words
        '''
        clist=[]
        for smiles in slist:
            smile=list(smiles)
            clist.append(smile)
        return clist

    @staticmethod
    def genDict():
        '''
         ==> words list and dicts
        '''
        #words list
        vocab=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h',
        'i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','-',
        '[',']','(',')','.','=','#','$',':','/','\\','@']
        word_to_idx = {word: i+1 for i, word in enumerate(vocab)} #  word - index dict
        word_to_idx['?'] = 0                                      #       unknown word
        idx_to_word={i+1: word for i, word in enumerate(vocab)}   # index -  word dict
        idx_to_word[0] = '?'
        with open("./tmp/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(word_to_idx,f)
        with open("./tmp/indexToWord.json","w",encoding="utf-8") as f:
            json.dump(idx_to_word,f)
        return word_to_idx,idx_to_word

    @staticmethod
    def wToIdx(clist,word_to_idx):
        '''
        smiles ==> index 
        '''
        features=[]
        for smiles in clist:
            feature=[]
            for s in smiles:
                if s in word_to_idx:
                    feature.append(int(word_to_idx[s]))
                else:
                    feature.append(0)
            features.append(feature)
        return features


    @staticmethod
    def pad(features,maxlen=200,PAD=0):
        '''
        fix length of every molecules to 100
          > 100 : cut off
          < 100 : add   0         
        '''
        padded_features=[]
        for feature in features:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
            padded_features.append(padded_feature)
        return padded_features


    def eval_each(self,modelname=None,path=None,chemspace="B3LYP_6-31g",predspace="B3LYP_6-31g",write=False,coeff=1.0):
        '''
        testing models -- one-by-one
        '''

        data_each=[]
        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]       

        icount=0
        # The used validing suits            
        tmp1="valid-tmp"+chemspace+predspace+"LSTM"
        with open(tmp1,'w') as ftmp:
           for suit in self.suits2:
              #print(suit)
              with open(suit,'r') as fsuits:
                 for line in fsuits:
                    icount=icount+1  
                    ftmp.write(line)
              print(suit, " : ", icount)

        print("Total molecules in validing suit : ", icount)          

        if path is None:
            path=['data/tes_'+self.chemspace+'.txt']
        tra_size=self.config.tra_size
        
        if modelname is None:
            modelname=self.modelloc
        
        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return

        print('The performance correction coefficient : ',coeff)

        model=torch.load(modelname)
        model=model.to(self.device)
        model.eval()
        
        basisnums,times,slist,names=LstmTool.readData(path,self.sdf_dir,tra_size,self.target,basis=basis)

        clist=LstmTool.seg(slist)
        with open('./tmp/wordToIndex.json','r',encoding='utf8') as f:
            word_to_idx=json.load(f)
        features=LstmTool.wToIdx(clist,word_to_idx)
        padded_features=LstmTool.pad(features)
        eval_features=torch.tensor(padded_features)
        eval_basis=torch.tensor(basisnums)
        eval_time=torch.tensor(times)
        #eval_names=torch.tensor(names)
        #nameDic={idx:name for idx,name in enumerate(names)}
        #eval_names=torch.tensor(nameDic.keys())

        eval_set=torch.utils.data.TensorDataset(eval_features,eval_basis,eval_time)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=self.config.batch_size,shuffle=False)

        preds=[]
        with torch.no_grad():

            with open(tmp1,'r') as ftmp:
                ftmplines=ftmp.readlines()

            err_mean=0.0
            errs=[]
            j=0

            ae=0.0
            for feature,basisnum,time in eval_iter:
                #j=0
                feature=feature.to(self.device)
                basisnum=basisnum.to(self.device)

                result=model(feature,basisnum)*coeff
                result=result.to('cpu')
                resultlist=result.numpy().tolist()
                preds.extend(resultlist)
                basislist=basisnum.to('cpu').numpy().tolist()
                timelist=time.numpy()
                timelist=timelist.tolist()
                print("len(resultlist)",len(resultlist),"len(ftmplines)",len(ftmplines)) 
                for i in range(len(resultlist)):
                    print("基组: %d, 预测值: %.5f, 预测值(corrected): %.5f, 真实值: %.5f" %(basislist[i],resultlist[i],resultlist[i],timelist[i]))
                    j+=1
                    #print("i : ", i)
                    #print("resultlist[i] : ", resultlist[i]) 
                    err1=(float(resultlist[i])-float(timelist[i]))/float(timelist[i])
                    print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',basislist[i],' real time : ',timelist[i],' predicted time: ',resultlist[i], 'err', err1)
                    data_each.append([ftmplines[i].split()[4],basislist[i],timelist[i],resultlist[i],err1])
                    single_err=abs((resultlist[i]-timelist[i])/timelist[i])
                    ae=ae+abs(resultlist[i]-timelist[i]) 
                    errs.append(single_err)
                    err_mean+=single_err
            err_mean=err_mean/j
            mae=ae/j
            errs=np.array(errs)
            variance=errs.var() 

            print("MRE：%.4f, MAE: %.4F, 方差: %.4f"%(err_mean,mae,variance))


        if write is True:
            if tra_size<4000:
                wbdir='results_'+str(tra_size)+'/'
            else:
                wbdir='results/'
            
            wbname=wbdir+'tes_lstm_'+self.chemspace+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})
            worksheet=workbook.add_worksheet("")
            #k=1
            worksheet.write_row('A1',['name','nbasis','res','real'])
            for i in range(len(names)):
                content=[names[i],basisnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")
        
        return data_each



    def eval(self,modelname=None,path=None,chemspace="B3LYP_6-31g",write=False):
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
        tmp1="./tmp/valid-tmp"+chemspace+"LSTM"
        with open(tmp1,'w') as ftmp:
           for suit in self.suits2:
              #print(suit)
              with open(suit,'r') as fsuits:
                 for line in fsuits:
                    icount=icount+1  
                    ftmp.write(line)
              print(suit, " : ", icount)

        print("Total molecules in validing suit : ", icount)          

#        if path is None:
#            path=['data/tes_'+self.chemspace+'.txt']
        
#        if modelname is None:
#            modelname=self.modelloc
        
        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return

        model=torch.load(modelname)
        model=model.to(self.device)
        model.eval()
        
        bnum,bnums,times,slist,names=LstmTool.readData([tmp1],self.sdf_dir,tra_size,self.target,basis=basis)

        clist=LstmTool.seg(slist)
        with open('./tmp/wordToIndex.json','r',encoding='utf8') as f:
            word_to_idx=json.load(f)
        features=LstmTool.wToIdx(clist,word_to_idx)
        padded_features=LstmTool.pad(features)
        eval_features=torch.tensor(padded_features)
        eval_basis=torch.tensor(bnum)
        eval_basis_s = torch.tensor(bnums)
        eval_time=torch.tensor(times)
        #eval_names=torch.tensor(names)
        #nameDic={idx:name for idx,name in enumerate(names)}
        #eval_names=torch.tensor(nameDic.keys())

        eval_set=torch.utils.data.TensorDataset(eval_features,eval_basis,eval_basis_s,eval_time)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=self.config.batch_size,shuffle=False)

        preds=[]
        with torch.no_grad():

            with open(tmp1,'r') as ftmp:
                ftmplines=ftmp.readlines()

            err_mean=0.0
            errs=[]
            j=0

            ae=0.0
            for feature,basisnum,basisnums,time in eval_iter:
                #j=0
                feature=feature.to(self.device)
                basisnum=basisnum.to(self.device)
                basisnums=basisnums.to(self.device)

                result=model(feature,basisnum,basisnums)
                result=result.to('cpu')
                resultlist=result.numpy().tolist()
                preds.extend(resultlist)
                basislist=basisnum.to('cpu').numpy().tolist()
                basislist_s=basisnums.to('cpu').numpy().tolist()
                timelist=time.numpy()
                timelist=timelist.tolist()
                print("len(resultlist)",len(resultlist),"len(ftmplines)",len(ftmplines)) 
                for i in range(len(resultlist)):
                    #print("基组:", basislist[i])
                    #print(" 预测值: %.5f, 预测值(corrected): %.5f, 真实值: %.5f" %(resultlist[i],resultlist[i],timelist[i]))
                    j+=1
                    #print("i : ", i)
                    #print("resultlist[i] : ", resultlist[i]) 
                    err1=(float(resultlist[i])-float(timelist[i]))/float(timelist[i])
                    print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',basislist[i],'basis sum num',basislist_s[i],' real time : ',timelist[i],' predicted time: ',resultlist[i], 'err', err1)
                    single_err=abs((resultlist[i]-timelist[i])/timelist[i])
                    ae=ae+abs(resultlist[i]-timelist[i]) 
                    errs.append(single_err)
                    err_mean+=single_err
            err_mean=err_mean/j
            mae=ae/j
            errs=np.array(errs)
            variance=errs.var()
            '''
            import pdb
            pdb.set_trace() 
            '''
            print("MRE：%.4f, MAE: %.4F, 方差: %.4f"%(err_mean,mae,variance))


        if write is True:
            if tra_size<4000:
                wbdir='results_'+str(tra_size)+'/'
            else:
                wbdir='results/'
            
            wbname=wbdir+'tes_lstm_'+self.chemspace+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})
            worksheet=workbook.add_worksheet("")
            #k=1
            worksheet.write_row('A1',['name','nbasis','res','real'])
            for i in range(len(names)):
                content=[names[i],basisnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")
        
        return [err_mean,mae,variance] 
    
    def train(self,path=None,mol_size="small"):
        dft  = self.chemspace.split("_")[0]
        basis= self.chemspace.split("_")[1]

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

        if path is None:
           path=tmp1

        #读取原始数据
        if mol_size == "small":
            bnum,bnums,times,slist,names=LstmTool.readData([tmp1],self.sdf_dir,tra_size,self.target,basis=basis)
        elif mol_size == "middle":
            bnum,bnums,times,slist,names=LstmTool.readData([tmp2],self.sdf_dir,tra_size,self.target,basis=basis)
        else:
            bnum,bnums,times,slist,names=LstmTool.readData([tmp3],self.sdf_dir,tra_size,self.target,basis=basis) 
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
        bnum,bnums,times,slist,names=LstmTool.readData([tmp1],self.sdf_dir,tra_size,self.target,basis=basis)
        '''

        #对smiles分词
        clist=LstmTool.seg(slist)

        if not os.path.exists("./tmp/wordToIndex.json"):
            #生成词表并保存
            word_to_idx,idx_to_word=LstmTool.genDict()
        else:
            with open("./tmp/wordToIndex.json",'r') as f1:
                word_to_idx=json.load(f1)
            with open("./tmp/indexToWord.json",'r') as f2:
                idx_to_word=json.load(f2)

        #将smiles编码用索引表示
        features=LstmTool.wToIdx(clist,word_to_idx)

        #对索引表示的smiles编码进行裁剪或补零
        padded_features=LstmTool.pad(features)

        #划分训练集、验证集,并转为tensor格式
        ratio=self.config.tra_set_ratio
        data_length=len(padded_features)
        border=int(ratio*data_length)
        train_features=torch.tensor(padded_features[0:border])
        train_basis=torch.tensor(bnum[0:border])
        train_basis_s = torch.tensor(bnums[0:border])
        train_time=torch.tensor(times[0:border])

        valid_features=None
        valid_basis=None
        valid_basis_s = None
        valid_time=None
        if ratio<1:
            valid_features=torch.tensor(padded_features[border:])
            valid_basis=torch.tensor(bnum[border:])
            valid_basis_s=torch.tensor(bnums[border:])
            valid_time=torch.tensor(times[border:])

        #读取预训练词向量
        #wvmodel=gensim.models.KeyedVectors.load_word2vec_format("../word2Vec_2.bin",binary=True,encoding='utf-8')
        #wvmodel=gensim.models.Word2Vec.load("../word2Vec_2")

        #词嵌入过程，即用预训练的词向量对词表中的词进行编码
        vocab_size=len(word_to_idx.keys())
        weight=torch.zeros(vocab_size,100) #100为embed_size

        # for i in range(len(wvmodel.index2word)):
        #     try:
        #         index=int(word_to_idx[wvmodel.index2word[i]])#找到预训练词向量中的词在词表中的索引
        #         #print(index)
        #     except:
        #         continue
        #     #print(idx_to_word[str(index)])
        #     weight[index,:]=torch.from_numpy(wvmodel.get_vector(
        #         idx_to_word[str(index)])) #将该索引对应的word_embedding（权重）保存在weight中


        model=BiLSTM.BiLSTM(embed_dim=self.config.input_size,hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,num_layers=self.config.num_layers,weight=None)
       
        loss_function=nn.L1Loss()
        optimizer=optim.Adam(model.parameters(),lr=self.config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma = 0.8)

        train_set=torch.utils.data.TensorDataset(train_features,train_basis,train_basis_s,train_time)
        train_iter=torch.utils.data.DataLoader(train_set,batch_size=self.config.batch_size,shuffle=True)

        if valid_features is not None:
            valid_set=torch.utils.data.TensorDataset(valid_features,valid_basis,valid_basis_s,valid_time)
            valid_iter=torch.utils.data.DataLoader(valid_set,batch_size=self.config.batch_size,shuffle=True)

        #start trainning
        model.train()
        model.to(self.device)
        
        valid_interval=self.config.valid_interval
        tra_losses=[]
        tra_mres=[]
        tra_maes=[]
        val_losses=[]
        val_mres=[]
        val_maes=[]
        
        len_train_data=len(train_features)
        #print(" ==> self.folder_mod",self.folder_mod)
        if not os.path.exists(self.folder_mod):
            os.makedirs(self.folder_mod)

        err=1.0
        save_step=10
        minMre=100.0
        bestEpoch=0
        modelloc_tmp=self.folder_mod+'/'+ 'lstm_' + self.chemspace + '_tot_tmp.pkl'
        y = []
        y1 = []
        for epoch in range(1,self.config.tra_num_epochs+1):
            #start=systime.time()
            train_loss=0.0
            # if err<0.2:
            #     optimizer.param_groups[0]['lr']=0.0005
            j=0
            err=0.0
            mae=0.0
            for feature,basisnum,basisnums,time in train_iter:
                model.zero_grad()
                feature=feature.to(self.device)
                basisnum=basisnum.to(self.device)
                basisnums=basisnums.to(self.device)
                time=time.to(self.device)
                result=model(feature,basisnum,basisnums)

                loss=loss_function(result,time)
                loss.backward()
                optimizer.step()
                #scheduler.step()
                train_loss+=loss
                reslist=result.to('cpu').detach().numpy()
                reslist=reslist.tolist()
                timelist=time.to('cpu').detach().numpy()
                timelist=timelist.tolist()
                for i in range(len(reslist)):
                    j+=1
                    mae_single=abs(reslist[i]-timelist[i])
                    mae+=mae_single
                    err+=mae_single/timelist[i]
            err=err/j
            tra_mres.append(err)
            mae=mae/j
            tra_maes.append(mae)
            tra_losses.append(train_loss/len_train_data)
            y.append(err)
            y1.append(mae)
            #end=systime.time()
            #runtime=end-start
            #print("epoch: %d, train loss: %.4f, mre: %.4f, mae:%.4f, time: %.2f"%(epoch+1,train_loss,err,mae,runtime))
            print("epoch: %d, train loss: %.4f, mre: %.4f, mae:%.4f"%(epoch,train_loss,err,mae))

            if epoch%save_step==0:
                torch.save(model,modelloc_tmp)
                eval_res=self.eval(modelname=modelloc_tmp,chemspace=self.chemspace,path=self.suits2)
                if eval_res[0]<minMre:
                    torch.save(model,self.folder_mod+'/'+ 'lstm_' + self.chemspace + '_tot2.pkl')
                    minMre=eval_res[0]
                    bestEpoch=epoch

            #做验证
            # if ratio<1:
            #     len_valid_data=len(valid_features)
            #     with torch.no_grad():
            #         if epoch>0 and (epoch+1)%valid_interval==0:
            #             print(os.linesep)
            #             valid_loss=0
            #             k=0
            #             vmae=0.0
            #             vmre=0.0
            #             for vft,vbs,vt in valid_iter:
            #                 vft=vft.to(self.device)
            #                 vbs=vbs.to(self.device)
            #                 vt=vt.to(self.device)
            #                 vres=model(vft,vbs)
            #                 vloss=loss_function(vres,vt)
            #                 valid_loss+=vloss
            #                 vreslist=vres.to('cpu').numpy().tolist()
            #                 vtlist=vt.to('cpu').numpy().tolist()
            #                 for i in range(len(vreslist)):
            #                     k+=1
            #                     vmae_single=abs(vreslist[i]-vtlist[i])
            #                     vmae+=vmae_single
            #                     vmre+=vmae_single/vtlist[i]
            #             vmre/=k
            #             vmae/=k
            #             val_mres.append(vmre)
            #             val_maes.append(vmae)
            #             val_losses.append(valid_loss/len_valid_data)
            #             print("epoch: %d, valid loss: %.4f, valid_mre: %.4f, valid_mae:%.4f"%(epoch+1,valid_loss,vmre,vmae))
            #             print(os.linesep)            

            #     self.gen_learning_curve(tra_losses,tra_mres,tra_maes,val_losses,val_mres,val_maes)

        
        #modelloc='model/'+self.chemspace+'_lstm.pkl'
        #torch.save(model, self.modelloc)
         
        print("trainning done! best epoch is "+str(bestEpoch))
        print("trainning completed!")
        print("training done : keep the best model and delete the intermediate models")
        os.remove(modelloc_tmp)

        data_path = os.getcwd() + '/eps/lstm/size/improve1/mre/'
        data_path1 = os.getcwd() + '/eps/lstm/size/improve1/mae/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(data_path1):
            os.makedirs(data_path1)

        '''
        if self.chemspace == "B3LYP_6-31g":
            np.save(data_path1 + 'LSTM_B3LYP_6-31g', y1)
        elif self.chemspace == "B3LYP_6-31gs":
            np.save(data_path1 + 'LSTM_B3LYP_6-31gs', y1)
        elif self.chemspace == "B3LYP_6-31pgs":
            np.save(data_path1 + 'LSTM_B3LYP_6-31pgs', y1)
        else:
            np.save(data_path1 + 'LSTM_P38_631gss2', y)
        '''
        if mol_size == "small":
            np.save(data_path1 + 'LSTM_B3LYP_6-31pgs_small', y1)
        elif mol_size == "middle":
            np.save(data_path1 + 'LSTM_B3LYP_6-31pgs_middle', y1)
        else:
            np.save(data_path1 + 'LSTM_B3LYP_6-31pgs_large', y1)

        return minMre    


    def finetune(self,epochs=30,lr=0.001,fc_dim1=10,fc_dim2=5,path=None):
        '''
        对模型做微调
        epoch: 训练的轮数
        lr: 学习率
        fc_dim1: 倒数第二层全连接层的输入维度
        fc_dim2: 最后一层全连接层的输入维度
        path: 训练集路径，一般无需手动指定
        '''
        if path is None:
            path=['./data/tra_'+self.chemspace+'.txt']
        tra_size=self.config.tra_size
        basisnums,times,slist,names=LstmTool.readData(path,self.sdf_dir,tra_size,self.target)

        #对smiles分词
        clist=LstmTool.seg(slist)

        if not os.path.exists("./tmp/wordToIndex.json"):
            #生成词表并保存
            word_to_idx,idx_to_word=LstmTool.genDict()
        else:
            with open("./tmp/wordToIndex.json",'r') as f1:
                word_to_idx=json.load(f1)
            with open("./tmp/indexToWord.json",'r') as f2:
                idx_to_word=json.load(f2)

        #将smiles编码用索引表示
        features=LstmTool.wToIdx(clist,word_to_idx)

        #对索引表示的smiles编码进行裁剪或补零
        padded_features=LstmTool.pad(features)

        #转化成tensor格式
        train_features=torch.tensor(padded_features)
        train_basis=torch.tensor(basisnums)
        train_time=torch.tensor(times)

        #读取预训练词向量
        wvmodel=gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin",binary=True,encoding='utf-8')

        #词嵌入过程，即用预训练的词向量对词表中的词进行编码
        vocab_size=len(word_to_idx.keys())
        weight=torch.zeros(vocab_size,100) #100为embed_size

        for i in range(len(wvmodel.index2word)):
            try:
                index=int(word_to_idx[wvmodel.index2word[i]])#找到预训练词向量中的词在词表中的索引
            except:
                continue
            weight[index,:]=torch.from_numpy(wvmodel.get_vector(
                idx_to_word[str(index)])) #将该索引对应的word_embedding（权重）保存在weight中

        if tra_size<4000:
            modeldir='model_'+str(tra_size)+'/'
        else:
            modeldir='model/'
        modelloc=modeldir+self.chemspace+'_lstm.pkl'
        model=torch.load(modelloc)   
        for param in model.parameters():
            param.requires_grad=False
        model.fc4=nn.Linear(fc_dim1,fc_dim2)
        model.fc5=nn.Linear(fc_dim2,1)
        model=model.to(self.device)
        model.train()

        train_set=torch.utils.data.TensorDataset(train_features,train_basis,train_time)
        train_iter=torch.utils.data.DataLoader(train_set,batch_size=self.config.batch_size,shuffle=True)

        loss_function=nn.MSELoss()#均方差
        optimizer=optim.Adam(model.parameters(), lr=lr)


        err=1.0
        for epoch in range(self.config.tra_num_epochs):
            start=systime.time()
            train_loss=0.0
            j=0
            err=0.0
            for feature,basisnum,time in train_iter:
                model.zero_grad()
                feature=feature.to(self.device)
                basisnum=basisnum.to(self.device)
                time=time.to(self.device)
                result=model(feature,basisnum)

                loss=loss_function(result,time)
                loss.backward()
                optimizer.step()
                train_loss+=loss
                reslist=result.to('cpu').squeeze(dim=1).detach().numpy()
                reslist=reslist.tolist()
                timelist=time.to('cpu').detach().numpy()
                timelist=timelist.tolist()
                for i in range(len(reslist)):
                    j+=1
                    err+=abs((reslist[i]-timelist[i])/timelist[i])
            err=err/j
            end=systime.time()
            runtime=end-start
            print("epoch: %d, train loss: %.4f, err: %.4f, time: %.2f"%(epoch,train_loss,err,runtime))

        torch.save(model, modeldir+self.chemspace+'_lstm_f.pkl')
        print("trainning completed!")    


            


    def predict(self,pdata,batch_size):#pdata:[[sdf1,sdf2,...],[nbasis1,nbasis2,...]],sdf1表示该sdf文件的路径
        '''
        调用模型进行推断
        pdata: 待预测数据，形式为[[sdf1,sdf2,...],[nbasis1,nbasis2,...]],sdf1表示该sdf文件的路径
        batch_size: batch大小
        '''
        model=torch.load('model/'+self.chemspace+'_lstm.pkl')
        model.eval()
        model=model.to(self.device)

        preds=[]

        slist=LstmTool.sdf_to_smiles(pdata[0])
        basisnums=pdata[1]
        if not isinstance(basisnums[0],float):
            basisnums=map(float,basisnums)

        clist=LstmTool.seg(slist)
        with open('./tmp/wordToIndex.json','r',encoding='utf8') as f:
            word_to_idx=json.load(f)
        features=LstmTool.wToIdx(clist,word_to_idx)
        padded_features=LstmTool.pad(features)

        
        eval_features=torch.tensor(padded_features)#.to(self.device)
        eval_basis=torch.tensor(basisnums)#.to(self.device)
        #eval_time=torch.tensor(times)
        #eval_names=torch.tensor(names)

        eval_set=torch.utils.data.TensorDataset(eval_features,eval_basis)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=batch_size,shuffle=False)
        

        with torch.no_grad():
            for feature,basisnum in eval_iter:
                feature=feature.to(self.device)
                basisnum=basisnum.to(self.device)

                result=model(feature,basisnum)
                result=result.to('cpu').squeeze(dim=1)
                resultlist=result.numpy().tolist()
                preds.extend(resultlist)
        
        return preds



            









                









