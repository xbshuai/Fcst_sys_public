import os
import os.path as osp
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import dgl
from dgl.data.utils import download
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import basis_set_exchange as bse
import Magnification

"""
本模块用于封装图模型(MGCN与MPNN)的数据集,
基于Tencent Alchemy Tools修改(https://github.com/tencent-alchemy/Alchemy)
"""

#该类的实例代表一个batch
class AlchemyBatcher:
    def __init__(self, graph=None,basisnum=None ,basisnums = None, label=None):
        '''
        构造方法
        graph: 以图数据结构表示的分子特征集合
        basisnum: 每个原子的基函数数目集合
        label: 预测目标(时间)集合
        '''
        self.graph     = graph
        self.basisnum  = basisnum
        self.basisnums = basisnums
        self.label     = label

#用于生成batch，返回一个AlchemyBatcher实例
def batcher():
    def batcher_dev(batch):
        graphs, basisnum, basisnums, labels = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        basisnum     = torch.stack(basisnum)
        basisnums   = torch.stack(basisnums)
        labels       = torch.stack(labels, 0)
        return AlchemyBatcher(graph=batch_graphs, basisnum=basisnum, basisnums = basisnums, label=labels)
    return batcher_dev

# 该类的实例用于封装图模型的数据集
class TencentAlchemyDataset(Dataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    def alchemy_nodes(self, mol, bnum_q):
        """Featurization for all atoms in a molecule. The atom indices
        will be preserved.
        Args:
            mol : rdkit.Chem.rdchem.Mol
              RDKit molecule object
        Returns
            atom_feats_dict : dict
              Dictionary for atom features
        """
        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
            # atom_feats_dict['bnum'].append(torch.FloatTensor(bnum_q[u]))# add atom's basisnum
            atom_feats_dict['node_type'].append(atom_type)

            h_u = []
            h_u += [
                #int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
                int(symbol == x) for x in ['H','Li','Be','B', 'C', 'N', 'O', 'F']
                #int(symbol == x) for x in ['H','He','Li','Be','B', 'C', 'N', 'O', 'F','Ne']
            ]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)
            ]
            h_u.append(num_h)
            h_u.extend(bnum_q[u])
            
            atom_feats_dict['n_feat'].append(torch.FloatTensor(h_u))
            

        atom_feats_dict['n_feat'] = torch.stack(atom_feats_dict['n_feat'],dim=0)
        atom_feats_dict['pos'] = torch.stack(atom_feats_dict['pos'], dim=0)
        # atom_feats_dict['bnum'] = torch.stack(atom_feats_dict['bnum'], dim=0)
        atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])

        return atom_feats_dict

    def alchemy_edges(self, mol, self_loop=True):
        """Featurization for all bonds in a molecule. The bond indices
        will be preserved.
        Args:
          mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        Returns
          bond_feats_dict : dict
              Dictionary for bond features
        """
        bond_feats_dict = defaultdict(list)

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict['e_feat'].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE,
                              Chem.rdchem.BondType.TRIPLE,
                              Chem.rdchem.BondType.AROMATIC, None)
                ])
                bond_feats_dict['distance'].append(
                    np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict['e_feat'] = torch.FloatTensor(bond_feats_dict['e_feat'])
        bond_feats_dict['distance'] = torch.FloatTensor(bond_feats_dict['distance']).reshape(-1, 1)

        return bond_feats_dict

    def sdf_to_dgl(self, sdf_file, bnum, bnum_s, bnum_q, time,self_loop=False):
        """
        Read sdf file and convert to dgl_graph
        Args:
            sdf_file: path of sdf file
            self_loop: Whetaher to add self loop
        Returns:
            g: DGLGraph
            l: related labels
        """
        print("str(sdf_file) : ", str(sdf_file))
        sdf = open(str(sdf_file)).read()
        mol = Chem.MolFromMolBlock(sdf, removeHs=False)

        g = dgl.DGLGraph()

        # add nodes
        num_atoms = mol.GetNumAtoms()
        atom_feats = self.alchemy_nodes(mol,bnum_q)
        g.add_nodes(num=num_atoms, data=atom_feats)

        # add edges
        # The model we were interested assumes a complete graph.
        # If this is not the case, do the code below instead
        #
        # for bond in mol.GetBonds():
        #     u = bond.GetBeginAtomIdx()
        #     v = bond.GetEndAtomIdx()
        if self_loop:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms)],
                [j for i in range(num_atoms) for j in range(num_atoms)])
        else:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                    j for i in range(num_atoms)
                    for j in range(num_atoms) if i != j
                ])

        bond_feats = self.alchemy_edges(mol, self_loop)
        g.edata.update(bond_feats)
        bnm =torch.FloatTensor([bnum])
        bnm_s = torch.FloatTensor([bnum_s])
        # for val/test set, labels are molecule ID
        l = torch.FloatTensor([time]) #if self.mode == 'train' or self.mode=='valid' else torch.LongTensor([int(sdf_file.stem)])
        return (g, bnm, bnm_s,l)

    def __init__(self, mode='train', rootdir='./',suits='tmp1',chemspace='m062x_6-31G#',folder_sdf='./' ,transform=None,pdata=None,tra_size=4000, target=2):
        '''
        构造方法
        mode: 指定调用场景，值域为['train', 'valid','test','pred']
        rootdir: data目录的上级目录,默认为当前目录,data目录下存放数据集文件
        chemspace: 化学空间
        transform: 定义了对分子的图结构特征进行的操作,默认为None，无需指定
        pdata: 待预测数据,形式为二维列表[[sdf1,sdf2,...],[basisum1,basisnum2,...]],当mode为'test'或'pred'时需显式指定,默认为None
        tra_size: 训练集规模
        '''

        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]
        

        assert mode in ['train', 'valid','test','pred'] #"mode should be train/valid/test/pred"
        self.mode       = mode
        self.transform  = transform
        self.rootdir    = rootdir
        self.folder_sdf = folder_sdf
        self.suits      = suits
        self.chemspace  = chemspace
        self.target     = target
        print("GdataSet:     chemspace  is: ", self.chemspace)
        print("GdataSet:       rootdir  is: ", self.rootdir)
        print("GdataSet:    folder_sdf  is: ", self.folder_sdf)
        print("GdataSet: working suits  is: ", self.suits)
        print("GdataSet: targeting prop is: ", self.target)
        self.pdata=pdata #pdata代表待预测数据,pdata=[[sdf1,sdf2,...],[basisum1,basisnum2,...]]

        self._load(tra_size,basis)

    def _load(self,tra_size,basis="6-31g"):
        sdfs,bnum,bnum_s,times,self.graphs, self.basisnum,self.labels = [],[],[],[],[],[],[]
        self.sdfnames = []
        self.basisnums = []
        sdfnames      = []
        bnum_q = []
        target_file= self.suits

        print("target :", self.target)
        print("self.mode :", self.mode)
        print("folder_sdf : ",self.folder_sdf) 

        count=0
        if self.mode=='train':
            print("loading the training suits ",target_file) 
            for line in open(target_file,'r'):
                count+=1
                if count>tra_size:
                    break
                
                temp=line.strip(os.linesep).split()
                time=float(temp[self.target])#
                #import pdb
                #pdb.set_trace()
                #basvec = []

                for i in range(len(temp)):
                    if temp[i] == 'contracted':
                        basisnum_s1 = float(temp[i+2])
                        basisnum_p1 = float(temp[i+3])
                        basisnum_d1 = float(temp[i+4])
                        basisnum_f1 = float(temp[i+5])
                        basisnum_g1 = float(temp[i+6])
                        basisnum_h1 = float(temp[i+7].strip(']'))
                        
                    if temp[i] == 'contracted_per':
                        
                        j = (len(temp) - i - 2)/6
                        basvec = []
                        for n in range(int(j)):
                            if n == 0:
                                basisnum_s = float(temp[i+2+6*n].strip('[[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(',').strip(']'))
                            elif n == j-1:
                                basisnum_s = float(temp[i+2+6*n].strip('[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(']]'))
                            else:
                                basisnum_s = float(temp[i+2+6*n].strip('[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(',').strip(']'))
                            basisnum = [basisnum_s, basisnum_p, basisnum_d, basisnum_f, basisnum_g, basisnum_h]
                            basvec.append(basisnum)
                        break
                
                basisnum1 = [basisnum_s1, basisnum_p1, basisnum_d1, basisnum_f1, basisnum_g1, basisnum_h1]
                basisnums = float(temp[0])
                
                bnum.append(basisnum1)
                bnum_s.append(basisnums)
                bnum_q.append(basvec)

                sdfname=str(temp[4]).split('_')[0]
                loc=self.folder_sdf+'/'+sdfname+'.sdf'
                sdfs.append(loc)
                times.append(time)
                sdfnames.append(sdfname)
        
        if self.mode=='valid':
            #import pdb
            #pdb.set_trace()
            print("loading the validing suits ",target_file)
            for line in open(target_file,'r'):
                temp=line.strip(os.linesep).split()
                time=float(temp[self.target])#时间

                sdfname=str(temp[4]).split('_')[0]
                loc=self.folder_sdf+'/'+sdfname+'.sdf'
                #basvec = []

                for i in range(len(temp)):
                    if temp[i] == 'contracted':
                        basisnum_s1 = float(temp[i+2])
                        basisnum_p1 = float(temp[i+3])
                        basisnum_d1 = float(temp[i+4])
                        basisnum_f1 = float(temp[i+5])
                        basisnum_g1 = float(temp[i+6])
                        basisnum_h1 = float(temp[i+7].strip(']'))
                        
                    if temp[i] == 'contracted_per':

                        j = (len(temp) - i - 2)/6
                        basvec = []
                        for n in range(int(j)):
                            if n == 0:
                                basisnum_s = float(temp[i+2+6*n].strip('[[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(',').strip(']'))
                            elif n == j-1:
                                basisnum_s = float(temp[i+2+6*n].strip('[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(']]'))
                            else:
                                basisnum_s = float(temp[i+2+6*n].strip('[').strip(','))
                                basisnum_p = float(temp[i+3+6*n].strip(','))
                                basisnum_d = float(temp[i+4+6*n].strip(','))
                                basisnum_f = float(temp[i+5+6*n].strip(','))
                                basisnum_g = float(temp[i+6+6*n].strip(','))
                                basisnum_h = float(temp[i+7+6*n].strip(',').strip(']'))
                            basisnum = [basisnum_s, basisnum_p, basisnum_d, basisnum_f, basisnum_g, basisnum_h]
                            basvec.append(basisnum)
                        break
                
                basisnum1 = [basisnum_s1, basisnum_p1, basisnum_d1, basisnum_f1, basisnum_g1, basisnum_h1]
                basisnums = float(temp[0])
                
                bnum.append(basisnum1)
                bnum_s.append(basisnums)
                bnum_q.append(basvec)
                sdfs.append(loc)
                times.append(time)
                sdfnames.append(sdfname)

        if self.mode=='test' or self.mode=='pred':
            sdfs=self.pdata[0]
            bnum=self.pdata[1]
            bnum_s = self.pdata[2]
            for i in range(len(sdfs)):
                times.append(1)

        i=0
        for sdf_file in sdfs:
            #print("sdf_file",sdf_file)
            result = self.sdf_to_dgl(sdf_file,bnum[i],bnum_s[i], bnum_q[i],times[i])
            if result is None:
                continue
            self.graphs.append(result[0])
            self.basisnum.append(result[1])
            self.basisnums.append(result[2])

#            basisnum2=float(Magnification.getNbasis(basis,sdf_file))
#            self.basisnums2.append(basisnum2)

            self.labels.append(result[3])
            self.sdfnames.append(sdfnames[i])
#            print(sdfnames[i]) 
            i+=1
            if i%50 ==0 :
               print(i, " loaded")
        self.normalize()
        print("Totally ", len(self.graphs), " samples loaded!")

    def normalize(self, mean=None, std=None):  #标准化
        labels = np.array([i.numpy() for i in self.labels])
        if mean is None:
            mean = np.mean(labels, axis=0)
        if std is None:
            std = np.std(labels, axis=0)
        self.mean = mean
        self.std = std

    def __len__(self):  
        return len(self.graphs)

    def __getitem__(self, idx):  
        g,basisnum,basisnums, l = self.graphs[idx], self.basisnum[idx],self.basisnums[idx],self.labels[idx]
        if self.transform:
            g = self.transform(g)
        return g,basisnum,basisnums, l

if __name__ == '__main__':
    alchemy_dataset = TencentAlchemyDataset()
    device = torch.device('cpu')
    # To speed up the training with multi-process data loader,
    # the num_workers could be set to > 1 to
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(),
                                shuffle=False,
                                num_workers=0)

    for step, batch in enumerate(alchemy_loader):
        print("bs =", batch.graph.batch_size)
        print('feature size =', batch.graph.ndata['n_feat'].size())
        print('pos size =', batch.graph.ndata['pos'].size())
        print('edge feature size =', batch.graph.edata['e_feat'].size())
        print('edge distance size =', batch.graph.edata['distance'].size())
        print('label size=', batch.label.size())
        print(dgl.sum_nodes(batch.graph, 'n_feat').size())
        break


