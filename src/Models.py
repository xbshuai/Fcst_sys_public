import torch
import Configs
import os

def prepare(model):

   model=model.upper()

   if model == "MPNN": 
      import MpnnToolEVAL 
# 

   return   


def ModelLoad(model,BAK,ref_chemspace,QC_packages,Machines):

   model=model.upper()

   # modelName0=BAK+"/"+QC_packages+"_"+Machines+"_"+ref_chemspace
   modelName0 = BAK + "/" + model
   if model=="MPNN":       
      modelName1="/mpnn_"+ref_chemspace+"_tot.pkl"
      # modelName1 = "/mpnn_P38_631gss_tot.pkl"
   elif model=="MGCN":       
      modelName1="/mgcn_"+ref_chemspace+"_tot.pkl"
   elif model=="LSTM":       
      modelName1="/lstm_"+ref_chemspace+"_tot.pkl"
      # modelName1 = "/lstm_P38_631gss_tot.pkl"
   elif model=="RF":       
      modelName1="/rfmodel_tot"
      
   modelName=modelName0+modelName1
   print(modelName)

   return modelName


def TrainAndEval(TR_para,TR_dir,chemspace,folder_sdf,suits_train,suits_valid,setsDir,model="MPNN",res_log='res-tot.log'):

   # configurations for the training  
   config=Configs.Config(tra_size=TR_para[0], tra_num_epochs=TR_para[1], batch_size=TR_para[2], lr=TR_para[3], tra_set_ratio=TR_para[4], valid_interval=TR_para[5])

   aimming=2 # The 2nd parameter, i.e. total CPU times
   if model=="MPNN": 
      import MpnnToolTRAIN 
      
      tool=MpnnToolTRAIN.MpnnTool(chemspace=chemspace, config=config, suits1=suits_train, suits2=suits_valid, folder_sdf=folder_sdf, folder_mod=TR_dir, target=aimming)

      for i in range(3):
         if i == 0:
            tool.train(path=setsDir,mol_size = "small")
         elif i == 1:
            tool.train(path=setsDir,mol_size = "middle")
         else:
            tool.train(path=setsDir,mol_size = "large")
      '''
      tool.train(path=setsDir)
      '''

   elif model=="MGCN":
      import MgcnToolTRAIN 
      tool=MgcnToolTRAIN.MgcnTool(chemspace=chemspace, config=config, suits1=suits_train, suits2=suits_valid, folder_sdf=folder_sdf, folder_mod=TR_dir, target=aimming)
       
      for i in range(3):
         if i == 0:
            tool.train(path=setsDir,mol_size = "small")
         elif i == 1:
            tool.train(path=setsDir,mol_size = "middle")
         else:
            tool.train(path=setsDir,mol_size = "large")
      '''
      tool.train(path=setsDir)
      '''
   elif model=="LSTM":
      import LstmToolTRAIN 
      tool=LstmToolTRAIN.LstmTool(chemspace=chemspace, config=config, suits1=suits_train, suits2=suits_valid, folder_sdf=folder_sdf, folder_mod=TR_dir, target=aimming)


      for i in range(3):
         if i == 0:
            tool.train(path=setsDir,mol_size = "small")
         elif i == 1:
            tool.train(path=setsDir,mol_size = "middle")
         else:
            tool.train(path=setsDir,mol_size = "large")
      '''
      tool.train(path=setsDir)
      '''

   elif model=="RF":
      import RfToolTRAIN
      series=['alkane','branch','ring','PE']
      #tool= RfToolTRAIN.RfTool(chemspace=chemspace, config=config, suits1=suits_train, suits2=suits_valid, folder_sdf=folder_sdf, folder_mod=TR_dir, target=aimming)
      tool = RfToolTRAIN.RfTool(chemspace=chemspace, config=config,                       suits2=suits_valid, folder_sdf=folder_sdf, folder_mod=TR_dir, target=aimming)
      for i in range(len(series)):
         if series[i]=='ring':  # the ring type has two componments
            suits_train.append(setsDir+'/'+chemspace+'/ring_sub') 
         suits_train.append(setsDir+'/'+chemspace+'/'+series[i])
         if not os.path.exists(TR_dir+'/'+'rfmodel_tot'):
            os.mkdir(TR_dir+'/'+'rfmodel_tot')
         modelName=TR_dir+'/'+'rfmodel_tot'+'/'+chemspace+'_'+str(i+1)+'.pkl'
         tool.train(path=suits_train,moltype=i+1,modelloc=modelName)
         
      #eval_res_tot=tool.eval(path=suits_valid,modeldir=TR_dir)  

   ###################################################################################
   #   Updated later
   ###################################################################################
   #exit(0)
   #eval_res=tool.eval(modelname=modelName,path=setsDir)  # need activate

   #line2=chemspace+" TOT, mre: "+str(eval_res[0])+", mae: "+str(eval_res[1])+", var: "+str(eval_res[2])+"\n"
   #with open(res_log,'a+',encoding='utf-8') as f:
   #   f.write(line2)
   #print(line2)






   return 

