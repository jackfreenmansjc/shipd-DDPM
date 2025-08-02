#import tools

import csv
import numpy as np
import json
import math
import matplotlib.pyplot as plt
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
import multiprocessing as mp #多进程使用模块,用于并行处理
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
import sklearn.preprocessing as PP
import sys
import os
#sys.path.append('/home/ada/Documents/HullParameterization')
from HullParameterization import Hull_Parameterization as HP

import torch
import torch.nn as nn
import torch.nn.functional as F 
print("HP module loaded:", HP)
##########################################################################
#                       section1：构造训练用的数据集：输入参数DesVec，包含8万多个船型数据（以45参数表示的船型样本）
current_dir = os.path.dirname(os.path.abspath(__file__))
#Load in Volume Prediction for Now
DesVecName = 'Input_Vectors.csv'
YName = 'GeometricMeasures/Volume.csv'
DS_path = current_dir+'/'
X_LIMITS_path = DS_path + 'X_LIMITS.npy' #np.load('/home/ada/Documents/HullParameterization/HullDiffusion/Restructured_Dataset/X_LIMITS.npy')
X_LIMITS = np.load(X_LIMITS_path)
print(X_LIMITS.shape)
folder_roots = ['Constrained_Randomized_Set_1']#,'Constrained_Randomized_Set_2','Constrained_Randomized_Set_3','Diffusion_Aug_Set_1','Diffusion_Aug_Set_2']
DesVec = []
YVec = []
for i in range(0,len(folder_roots)):
    path = DS_path + folder_roots[i] + '/'    
    #Location of Design Vectors
    with open(path + DesVecName) as csvfile:
        reader = csv.reader(csvfile)
        for count, row in enumerate(reader):
            if count != 0:
                DesVec.append(row)

DesVec = np.array(DesVec)
DesVec = DesVec.astype(np.float32())
np.save('DesVec_82k.npy',DesVec)
#清除desvec中无用的参数，这些参数和船的球鼻艏和艉部鼻艏有关。
idx_BBFactors = [33,34,35,36,37]
idx_BB = 31

idx_SBFactors = [38,39,40,41,42,43,44]
idx_SB = 32

for i in range(0,len(DesVec)):
    
    DesVec[i,idx_BBFactors] = DesVec[i,idx_BB] * DesVec[i,idx_BBFactors] 
    DesVec[i,idx_SBFactors] = DesVec[i,idx_SB] * DesVec[i,idx_SBFactors]
#####################################################
#                       secton2：通过8w多个船型参数，计算每个船型对应的几何特性，Vol,WP,LCB,VCB,LCF,Ixx,Iyy,WSA,WL
#                       将船划分出101等份的水线，每条水线由1000个点构成
#                       结果作为样本的输出值Y，回归模型建立X(来自desec)和Y（此处是wsa湿表面积）的映射关系
def run_IMAP_multiprocessing(func, argument_list, chunksize = None, show_prog = True):
    """Run function in parallel
    Parameters
    ----------
    func:          function
                    Python function to run in parallel.
    argument_list: list [N]
                    List of arguments to be passed to the function in each parallel run.
            
    show_prog:     boolean
                    If true a progress bas will be displayed to show progress. Default: True.
    Returns
    -------
    output:        list [N,]
                    outputs of the function for the given arguments.
    """
    #Reserve 2 threads for other Tasks
    #pool = mp.Pool(processes=mp.cpu_count()-2)
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list,chunksize=chunksize), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list,chunksize=chunksize):
            result_list_tqdm.append(result)
    return result_list_tqdm
def run_MAP_multiprocessing(func, argument_list, chunksize = None, show_prog = True):
    """Run function in parallel
    Parameters
    ----------
    func:          function
                    Python function to run in parallel.
    argument_list: list [N]
                    List of arguments to be passed to the function in each parallel run.
            
    show_prog:     boolean
                    If true a progress bas will be displayed to show progress. Default: True.
    Returns
    -------
    output:        list [N,]
                    outputs of the function for the given arguments.
    """
    #Reserve 2 threads for other Tasks
    #pool = mp.Pool(processes=mp.cpu_count()-2)
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.map(func=func, iterable=argument_list,chunksize=chunksize), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.map(func=func, iterable=argument_list,chunksize=chunksize):
            result_list_tqdm.append(result)

    return result_list_tqdm
def Calc_GeometricProperties(x):
    '''
    This function takes in a Ship Design Vector and calculates the volumetric properties of the hull 
    
    It returns the values for:
    
    Z / L             -> nondimensialized vector for the height at which each value was measured
    Volume / L^3
    Area of Waterplane / L^2
    Longitudinal Centers of Buoyancy/L
    Vertical Center of Buoyancy / L
    Longitudinal Center of Flotation / L
    Ixx / L^4
    Iyy / L^4
    
    where L = LOA of the design vector ( x[0])
    
    This function is written to be paralellized   
    
    '''
    hull = HP(x)
    Z = hull.Calc_VolumeProperties(NUM_WL = 101, PointsPerWL = 1000)
    L = x[0]
    z = np.divide(Z,L)
    Vol = np.divide(hull.Volumes,L**3.0)
    WP = np.divide(hull.Areas_WP,L**2.0)
    LCF = np.divide(hull.LCFs,L)
    Ixx = np.divide(hull.I_WP[:,0],L**4.0)
    Iyy = np.divide(hull.I_WP[:,1],L**4.0)
    LCB = np.divide(hull.VolumeCentroids[:,0],L)
    VCB = np.divide(hull.VolumeCentroids[:,0],L)
    WSA = np.divide(hull.Area_WS,L**2.0)
    WL = np.divide(hull.WL_Lengths,L)
    return np.concatenate((z,Vol,WP,LCB,VCB,LCF,Ixx,Iyy,WSA,WL),axis=0)
#Compute Y
# Run Multiprocessing to Calculate the Geometric Measures
CHUNKS = 256
print('Calculating Hulls...')
print('Threads: ' + str(mp.cpu_count()))
pool = mp.Pool(processes=mp.cpu_count()-2)
#Y = [Performance_Metric(DesVec[i]) for i in tqdm(range(0,len(DesVec)))]
#Y = run_MAP_multiprocessing(Calc_GeometricProperties, DesVec,chunksize=CHUNKS,show_prog=True)
#Y = np.array(Y)
Y = Calc_GeometricProperties(DesVec)
np.save('GeometricMeasures.npy',Y)
print('Hull Calculations Complete!')
###############################################################
#                              section3 建立回归模型
#                           这里需要注意的是，X包含了44个参数（45参数中除了第一次船长参数的剩下部分+WL水线参数），Y是湿表面积
#                           这样获得一个计算不同水线下船湿表面积的回归模型
class Regression_ResNet(torch.nn.Module):
    def __init__(self, Reg_Dict):
        nn.Module.__init__(self)
        self.xdim = Reg_Dict['xdim']
        self.ydim = 1
        self.tdim = Reg_Dict['tdim']
        self.net = Reg_Dict['net']
        self.fc = nn.ModuleList()
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i])) 
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        '''
        #self.tc = nn.ModuleList()
        #for i in range(0, len(self.net)):
            self.tc.append(self.LinLayer(self.tdim,self.net[i]))
        self.tc.append(self.LinLayer(self.tdim, self.tdim))
        '''
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.ydim))
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        #self.T_embed = nn.Linear(self.ydim, self.tdim)  
    def LinLayer(self, dimi, dimo):
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
    def forward(self, x):
        x = self.X_embed(x)
        res_x = x
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
        x = torch.add(x,res_x)
        x = self.finalLayer(x)
        return x
class Regressor_Training_Env:
    def __init__(self, Reg_Dict, DesVec, Y):
        self.Reg_Dict = Reg_Dict
        self.DesVec = DesVec
        self.QT = Data_Normalizer(X_LIMITS[:,0],X_LIMITS[:,1],len(DesVec))
        self.X = np.copy(DesVec[:,1:])
        # Quantile Transform X:
        self.X = self.QT.fit_Data(self.X)
        self.Y = np.copy(Y)
        self.model = Regression_ResNet(self.Reg_Dict)
        self.device =torch.device(self.Reg_Dict['device_name'])
        self.model.to(self.device)
        self.data_length = len(self.X)
        self.batch_size = self.Reg_Dict['batch_size']
        self.num_epochs = self.Reg_Dict['Training_Epochs']
        lr = self.Reg_Dict['lr']
        self.init_lr = lr
        weight_decay = self.Reg_Dict['weight_decay']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    '''
    ==============================================================================
    Base Regression Training Functions
    ==============================================================================
    '''
    def run_regressor_step(self,x,y):
        self.optimizer.zero_grad()
        ones = torch.ones_like(y)
        predicted_y = self.model(x)
        loss =  F.mse_loss(predicted_y, y)
        #print(loss)
        loss.backward()
        self.optimizer.step()
        return loss  
    
    def run_train_regressors_loop(self,batches_per_epoch=64, subsample_per_batch = 64, num_WL_Steps = 101):
            num_batches = self.data_length // self.batch_size
            batches_per_epoch = min(num_batches,batches_per_epoch)
            T_vec = np.linspace(0,1,num_WL_Steps)
            print('Regressor Model Training...')
            for i in tqdm(range(0,self.num_epochs)):
                for j in range(0,batches_per_epoch):
                    A = np.random.randint(0,self.data_length,self.batch_size)
                    x_batch = torch.tensor(self.X[A]).float().to(self.device) 
                    for k in range(0,subsample_per_batch):
                        #Random Waterline
                        t = np.random.randint(0,num_WL_Steps,(self.batch_size,))
                        t_tens = torch.tensor(T_vec[t,np.newaxis]).float().to(self.device)
                        #Interpolate Volume
                        Y_calc = np.array([HP.interp(y[i],T_vec,t[i]) for i in range(0,len(t))])
                        y = self.Y[A,t]
                        y_batch = torch.tensor(y[:,np.newaxis]).float().to(self.device)
                        x = torch.cat((x_batch,t_tens),dim=1)
                        loss = self.run_regressor_step(x,y_batch)
                if i % 1000 == 0:
                    print('Epoch: ' + str(i) + ' Loss: ' + str(loss))   
            print('Regression Model Training Complete!')
            self.model.eval()
            eval_size = 10000
            A = np.random.randint(0,self.data_length,eval_size)
            t = np.random.random((eval_size,1))
            t_tens = torch.tensor(t).float().to(self.device)
            x_eval = torch.tensor(self.X[A]).float().to(self.device)
            x_eval = torch.cat((x_eval, t_tens),dim=1) 
            Y_pred = self.model(x_eval)
            Y_pred = Y_pred.to(torch.device('cpu')).detach().numpy() 
            y = self.Y[A]
            Y_calc = np.array([HP.interp(y[i],T_vec,t[i]) for i in range(0,len(t))])
            Rsq = r2_score(Y_calc, Y_pred)
            print("R2 score of Y:" + str(Rsq))
    # SAVE FUNCTIONS  
    def load_trained_model(self):
        label = self.Reg_Dict['Model_Path']
        self.model.load_state_dict(torch.load(label))
        self.model.to(self.device)
    def Save_model(self,PATH):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
       
        '''
        #PATH = current_dir+'/'
        torch.save(self.model.state_dict(), PATH + self.Reg_Dict['Model_Label']+'.pth')
        JSON = json.dumps(self.Reg_Dict)
        f = open(PATH + self.Reg_Dict['Model_Label'] + '_Dict.json', 'w')
        f.write(JSON)
        f.close()
####################################################################
#                            section4 启动resnet模型训练
#Regression model Dict
nodes = 512

Reg_Dict = {
        'xdim' : len(DesVec[0])-1 + 1,              # Dimension of parametric design vector
        'ydim': 1,                              # trains regression model for each objective
        'tdim': nodes,                            # dimension of latent variable
        'net': [nodes,nodes,nodes],                       # network architecture        
        'Training_Epochs': 30000,               # number of training epochs
        'batch_size': 1024,                       # batch size
        'Model_Label': 'Regressor_WSA',         # labels for regressors
                    
        'lr' : 0.001,                          # learning rate
        'weight_decay': 0.0,                   # weight decay
        'device_name': 'cuda:0'}    


num_WL_Steps = 101

Y_set = np.log10(Y[:,8*num_WL_Steps:9*num_WL_Steps])
#Y_set = Y[:,8*num_WL_Steps:9*num_WL_Steps]
idx = np.where(np.isnan(Y_set))
print(idx)

Y_set[idx] = -6.0 #fix nan to dummy value

print(Y_set.shape)

REG = Regressor_Training_Env(Reg_Dict, DesVec,Y_set)

REG.run_train_regressors_loop(batches_per_epoch=8, subsample_per_batch = 8, num_WL_Steps = num_WL_Steps)

PATH1 = current_dir+'/'
REG.Save_model(PATH1)

################################################################################################
#                             section5 模型评估
REG.model.eval()

sample_size = 100000

T_vec = np.linspace(0,1,num_WL_Steps)
A = np.random.randint(0,len(DesVec),sample_size)

t = np.random.random((sample_size,1))
t_tens = torch.tensor(t).float().to(REG.device)

x_eval = torch.tensor(REG.X[A]).float().to(REG.device)

x_eval = torch.cat((x_eval, t_tens),dim=1) 

Y_pred = REG.model(x_eval)
Y_pred = Y_pred.to(torch.device('cpu')).detach().numpy() 

y = REG.Y[A]

Y_calc = np.array([HP.interp(y[i],T_vec,t[i]) for i in range(0,len(t))])

#MAEP = np.mean(np.abs(np.power(10,Y_calc)-np.power(10,Y_pred)/np.power(10,Y_calc)))
MAEP = np.mean(np.abs(Y_calc-Y_pred)/np.abs(Y_calc))

Y_scaled_calc = 10**Y_calc
Y_scaled_pred = 10**Y_pred  


print('Log scale MAEP: ' + str(MAEP*100.0) + '%')

MAEP_scaled = np.mean(np.abs(Y_scaled_calc-Y_scaled_pred)/np.abs(Y_scaled_calc))
print('Scaled MAEP: ' + str(MAEP_scaled*100.0) + '%')
Rsq = r2_score(Y_scaled_calc, Y_scaled_pred)

print("R2 score of Scaled WSA Prediction: " + str(Rsq))
