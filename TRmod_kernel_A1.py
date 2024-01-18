import sys
import os
import time
import socket

hostname = socket.gethostname()
PWD=os.getcwd()
SRC=PWD+"/src"
RAW=PWD+"/database/rawdata"
BAK=PWD+"/database/trained-models"

# add the runtime environments
print(SRC)
sys.path.append(SRC)

# originally import 
import torch
import Models
import Configs
#import PredictTime
#import Magnification
#import DecideRefSpace

# rdkit for chem-informatics
#from rdkit import Chem
#from rdkit.Chem import AllChem

# ==> parameters to be used (IO later) 
# ML models, related
ML_models   =  ["LSTM"]
TR_para     =  [25,350,50,0.01,1.0,2] # [NtrainSet,Nepoch,BatchSize,LRstep,TrainRatio,ValidInt]
TRM_dir      = PWD + "/database/training-models"
# SDFs and Crawled folder, related
sdfsH       = RAW + "/Arxiv1911.05569v1_sdfs_H"
setsDir     = RAW + "/G09data.3D"
# Functionals and basis sets, related
#functionals = ['B3LYP','bhandhlyp','BLYP','CAM-B3LYP','LC-BLYP','M06','M062x','PBE1PBE','wb97xd']
# bases       = ['6-31g','6-31gs','6-31pgs']
functionals = ['B3LYP']
bases       = ['6-31pgs']

# training/validing/testing sets
# ==> training sets, manual selections for adjusting the training models
suits_train = []
#suits_train.append("branch")
#suits_train.append("ring")
#suits_train.append("ring_sub")
#suits_train.append("alkane")
#suits_train.append("PE")
suits_train.append("Gaussian_inputs_training2")
suits_train.append("Gaussian_inputs_training")
suits_train.append("Gaussian_inputs_training3")
suits_train.append("Gaussian_inputs_training4")
suits_train.append("Gaussian_inputs_training5")

# ==> validing/testing sets, manual selections for adjusting 
suits_valid = []
suits_valid.append("Gaussian_inputs_validing")
suits_valid.append("Gaussian_inputs_validing2")
#suits_valid.append("Gaussian_inputs_testing")
#suits_valid.append("Gaussian_inputs_testing2")

# Training and validing/testing process
for mod in ML_models:    # models
   #Models.prepare(mod)
   for funct in functionals:
      for basis in bases:
         chemspace=funct+'_'+basis

         # generate the full path for the training and validing sets
         train_tmp=[]
         for i in range(len(suits_train)):
            tmp = setsDir + "/" + chemspace + "/" + suits_train[i]
            train_tmp.append(tmp)
         valid_tmp=[]   
         for i in range(len(suits_valid)):
            tmp = setsDir + "/" + chemspace + "/" + suits_valid[i]
            valid_tmp.append(tmp)

         # Mkdir the "training" folder for usage
         TR_dir = TRM_dir + "/" + "G09_ERA_" + funct + "_" + basis
         if not os.path.exists(TR_dir):
            os.mkdir(TR_dir)

         # Training and evaluating
         Models.TrainAndEval(TR_para=TR_para,TR_dir=TR_dir,chemspace=chemspace,folder_sdf=sdfsH,suits_train=train_tmp,suits_valid=valid_tmp,setsDir=setsDir,model=mod)


print("All the models have been trained evaluated")

exit(0)

