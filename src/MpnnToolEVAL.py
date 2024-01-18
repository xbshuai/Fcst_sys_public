import argparse
import torch as th
import torch.nn as nn
import numpy as np
from MPNN import MPNNModel
from torch.utils.data import DataLoader
from GDataSetEVAL import TADataset, batcher, get_index
import os
import xlsxwriter
from ModelTool import ModelTool

th.manual_seed(2)

import basis_set_exchange as bse
from Magnification import getNbasis, getNbasis_noRDkit


# MPNN模型的工具类
class MpnnTool(ModelTool):
    
    def __init__ (self,chemspace,config,suits1,sdf_dir,target):
        '''
        construction
        chemspace
        config :  need Configs.Config to initialize
        '''
        ModelTool.__init__(self,chemspace,config,sdf_dir,target)
        self.suits1=suits1

    def evalsuit(self, modelname, chemspace="B3LYP_6-31g", path='./', write=False):
        '''
        predicting the timing basing on the specified model
        path: path of testing suits; default as 'data/tes_'+self.chemspace+'.txt'
        write: check whether into xlsx format
        '''

        dft = chemspace.split("_")[0]
        basis = chemspace.split("_")[1]

        tra_size = self.config.tra_size
        print("self.suits1  : ", self.suits1)
        print("self.sdf_dir : ", self.sdf_dir)

        names = []
        mollist = []
        baslist = []
        basnumlist = []

        for isuit in self.suits1:
            imol = self.sdf_dir + "/" + isuit
            # obasis, nbasis = getNbasis(bas=basis, sdf=imol)
            obasis, nbasis = getNbasis_noRDkit(bas=basis, sdf=imol)
            # ibas = getNbasis_noRDkit(bas=basis,sdf=imol)
            print("imol : ", imol, " ibas : ", nbasis)
            mollist.append(imol)
            baslist.append(nbasis)
            names.append(isuit)
            basnumlist.append(obasis)

            # print(" mollist ", mollist )
        # print(" baslist ", baslist )

        pdata = [mollist, basnumlist, baslist]

        dataset = TADataset(mode='pred', rootdir=path, chemspace=self.chemspace, folder_sdf=self.sdf_dir, pdata=pdata,
                            tra_size=tra_size, target=self.target)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.config.batch_size,
                            collate_fn=batcher(),
                            shuffle=False,
                            num_workers=0)

        if not os.path.exists(modelname):
            print(modelname + " does not exist!")
            return
        model = th.load(modelname, map_location=self.device)
        # model.to(self.device)

        model.eval()
        bnums = []
        bnums_s = []
        times = []
        preds = []

        with th.no_grad():
            err = 0
            errs = []
            j = 0
            mae = 0.0
            for idx, batch in enumerate(loader):
                batch.graph = batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum = batch.basisnum.to(self.device)
                batch.basisnums = batch.basisnums.to(self.device)
                res = model(batch.graph,batch.basisnum,batch.basisnums)
                # batch.sdf=batch.basisnum.to(self.device)
                res = res.to('cpu')
                # mae = MAE_fn(res, batch.label)
                # w_mae += mae.detach().item()
                reslist = res.numpy()
                # print("reslist  : ",reslist)
                reslist = res.tolist()
                batch.label = batch.label.to('cpu')
                batch.basisnum = batch.basisnum.to('cpu')
                batch.basisnums = batch.basisnums.to('cpu')
                timelist = batch.label.numpy()
                # print("timelist : ",timelist)
                timelist = timelist.tolist()
                bnumlist = batch.basisnum.numpy().tolist()
                bnumslist = batch.basisnums.numpy().tolist()

                for i in range(len(reslist)):
                    # print(i, " ===> initial < === ", reslist[i])
                    time = timelist[i][0]
                    ares = reslist[i]
                    bnum = bnumlist[i][0]
                    bnum_s = bnumslist[i][0]

                    # print(bnum)
                    times.append(time)
                    preds.append(ares)
                    bnums.append(bnum)
                    bnums_s.append(bnum_s)
                    # sdflist.append(sdf)
                    # print('i: ',i, ' sdf/mol: ', mollist[i],' basis num: ',bnum,' real time : ',time,' predicted time: ',ares)

        len_names = len(names)
        not_in_names = get_index()
        for i in range(len_names):
            if i in not_in_names:
                names.pop(i)
        i = 0
        print("len(names)", len(names), "len(preds)", len(preds))
        dest = os.getcwd() + "/P38_MPNN_631gss1_augV"
        destData = open(dest, "w", encoding='utf-8')
        for isuit in self.suits1:
            destLineData = str(i + 1) +  " " + names[i] + " " + str(preds[i])
            print(destLineData)
            destData.write(destLineData + '\n')
            i = i + 1
            if i >= len(names):
                break


    def eval(self,modelname,chemspace="B3LYP_6-31g",path='./',mol="sample.sdf",write=False):
        '''
        testing model
        path: data upper folder, default folder, data with data
        write:  can be xlsx file
        '''
        dft      = chemspace.split("_")[0]
        basis    = chemspace.split("_")[1]
        #print("dft ", dft ,"  basis", basis)        
        tra_size = self.config.tra_size
        molecule = path+"/"+mol
        obasis, nbasis = getNbasis(bas=basis,sdf=molecule)
        print(" nbasis ", nbasis)
        pdata=[[molecule],[obasis], [nbasis]]
        #exit(0)

        #dataset = TencentAlchemyDataset(mode='valid',rootdir=path,chemspace=self.chemspace,tra_size=tra_size)
        dataset=TADataset(mode='test',rootdir=path,suits=molecule,chemspace=chemspace,pdata=pdata,tra_size=tra_size,target=self.target)

        loader = DataLoader(dataset     = dataset,
                            batch_size  = self.config.batch_size,
                            collate_fn  = batcher(),
                            shuffle     = False,
                            num_workers = 0)

        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return
        #model = th.load(modelname,map_location=th.device('cpu'))
        model = th.load(modelname)
        model.to(self.device)

        #loss_fn = nn.MSELoss()
        #MAE_fn = nn.L1Loss() 
        model.eval()
        bnums = []
        bnums_s = []
        times = []
        preds = []
        # sdflist = []

        with th.no_grad():
            err=0
            errs=[]
            j=0
            mae=0.0
            #with open(tmp1,'r') as ftmp:
            #    ftmplines=ftmp.readlines()
            #print(ftmplines)   

            for idx,batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                batch.basisnums = batch.basisnums.to(self.device)
                #batch.sdf=batch.basisnum.to(self.device)
                res = model(batch.graph,batch.basisnum,batch.basisnums)
                res=res.to('cpu')
                #mae = MAE_fn(res, batch.label)
                #w_mae += mae.detach().item()
                #reslist=res.numpy()
                #print(reslist)
                reslist=res.tolist()
                batch.label=batch.label.to('cpu')
                batch.basisnum=batch.basisnum.to('cpu')
                batch.basisnums = batch.basisnums.to('cpu')
                timelist=batch.label.numpy()
                #print(timelist)
                timelist=timelist.tolist()
                bnumlist = batch.basisnum.numpy().tolist()
                bnumslist = batch.basisnums.numpy().tolist()

                for i in range(len(reslist)):
                    #print(i, " ===> initial < === ", reslist[i])
                    time = timelist[i][0]
                    ares = reslist[i]
                    bnum = bnumlist[i][0]
                    bnum_s = bnumslist[i][0]

                    # print(bnum)
                    times.append(time)
                    preds.append(ares)
                    bnums.append(bnum)
                    bnums_s.append(bnum_s)
                    #sdflist.append(sdf) 
                    err1=(float(time)-float(ares))/float(time)
                    #print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',bnum,' real time : ',time,' predicted time: ',ares, 'err', err1)
                    ae=abs(time-ares)
                    single_err=ae/time
                    err+=single_err
                    mae+=ae
                    errs.append(single_err)
                    j+=1
            # err_mean=err/j
            # mae/=j
            # errs=np.array(errs)
            # variance=errs.var()
            # print("MRE：%.4f, MAE: %.4F, VAR: %.4f"%(err_mean,mae,variance))

        if write==True:
            wbname='results_'+self.chemspace+'/'+'tes_mpnn_'+str(tra_size)+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})# benchmark results
            worksheet=workbook.add_worksheet("")
            worksheet.write_row('A1',['nbasis','res','real'])
            for i in range(len(bnums)):
                content=[bnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")

        return reslist

#        return [err_mean,mae,variance]    

