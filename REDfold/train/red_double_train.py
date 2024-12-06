import numpy as np
import subprocess
import collections
import random
import time
import sys
import os
import argparse
from tqdm import tqdm
import csv

from scipy.sparse import diags
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append('./..')

from utils.utils import *
from data_generator.data_generator import RNASSDataGenerator
from data_generator.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from model.models import FCDenseNet
# from utils.postprocess import postprocess_orig, postprocess_proposed

from shutil import copyfile

#from network import FCDenseNet as Model
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from operator import add

# 모델과 텐서를 모두 더블로 변경하는 부분 추가
torch.set_default_dtype(torch.float64)

def f1_loss(pred_a, true_a, device=0, eps=1e-11):
    device = torch.device("cuda:{}".format(device))
    pred_a  = -(F.relu(-pred_a+1)-1).to(device)

    true_a = true_a.unsqueeze(1).to(device)
    unfold = nn.Unfold(kernel_size=(3, 3), padding=1)
    true_a_tmp = unfold(true_a)
    w = torch.Tensor([0, 0.0, 0, 0.0, 1, 0.0, 0, 0.0, 0]).to(device)
    true_a_tmp = true_a_tmp.transpose(1, 2).matmul(w.view(w.size(0), -1)).transpose(1, 2)
    true_a = true_a_tmp.view(true_a.shape)
    true_a = true_a.squeeze(1)

    tp = pred_a * true_a
    tp = torch.sum(tp, (1,2))

    fp = pred_a * (1 - true_a)
    fp = torch.sum(fp, (1,2))

    fn = (1 - pred_a) * true_a
    fn = torch.sum(fn, (1,2))

    f1 = torch.div((2*tp + eps), (2*tp + fp + fn + eps))
    return (1 - f1.mean()).to(device)

def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def seqout(y1,L1,seqhot):
    seq1 = ""
    fast = ""
    y1 = y1.float()

    for a1 in range(L1):
        Id1 = np.nonzero(seqhot[0, a1]).item()
        seq1 += BASE1[Id1]

        Id2 = np.nonzero(y1[0, a1, :L1])
        if (Id2.nelement()):
            fast += '(' if (a1 < Id2) else ')'
        else:
            fast += '.'
    seq1 += "\n"

def get_ct_dict(predict_matrix, batch_num, ct_dict):
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix, batch_num, ct_dict, dot_file_dict, seq_embedding, seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis=1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    letter = 'AUCG'
    ct_dict[batch_num] = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]    
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name, seq_letter, dot_list[:len(seq_letter)])]
    return ct_dict, dot_file_dict

def evaluate_exact_a(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum().double()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum().double()
    true_p = torch.Tensor(true_a).sum().double()
    fp = pred_p - tp
    fn = true_p - tp
    tn_map = torch.sign(torch.Tensor(pred_a) * (1 - torch.Tensor(true_a)))
    tn = tn_map.sum().double()
    
    sensitivity = (tp + eps) / (tp + fn + eps)
    positive_predictive_value = (tp + eps) / (tp + fp + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    
    return positive_predictive_value, sensitivity, f1_score, accuracy

def sinkhorn(m, x, max_iter, threshold, lamda, eps, device_num):
    device = torch.device('cuda:{}'.format(device_num))
    
    mask = constraint_matrix_sinkhorn(m, x, process_device=device_num)

    cost = -(m * mask).to(device)

    b = m.size(0)
    d = m.size(1)
    r = torch.ones(d).repeat(b, 1).to(device) / d # b x d tensor
    c = torch.ones(d).repeat(b, 1).to(device) / d # b x d tensor

    k = (-lamda * cost).exp().to(device) # b x d x d tensor
    u = torch.ones(b, d, 1).to(device) # b x d x 1 tensor
    eye = torch.eye(d).repeat(b, 1, 1).to(device) / d # b x d x d tensor
    k_tilde = torch.bmm(eye, k) # b x d x d tensor
    for i in range(max_iter):
        pre_u = u
        u = 1.0 / torch.bmm(k_tilde, (c / torch.bmm(k_tilde, u).squeeze(-1)).unsqueeze(-1))
        if (u - pre_u).abs().sum(-1).mean().item() < threshold:
            break
    v = (c / torch.bmm(k_tilde, u).squeeze(-1)).unsqueeze(-1).to(device) # b x d x 1

    diag_u = torch.diag_embed(u.squeeze(-1), offset=0, dim1=-2, dim2=-1).to(device) # b x d x d
    diag_v = torch.diag_embed(v.squeeze(-1), offset=0, dim1=-2, dim2=-1).to(device) # b x d x d
    out = torch.bmm(torch.bmm(diag_u, k), diag_v)
    
    out = (1-2*eps)*out + eps
    
    return out

def constraint_matrix_sinkhorn(u, x, process_device=0):
    device = torch.device('cuda:{}'.format(process_device))

    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    base_ag = base_a + base_g
    base_uc = base_u + base_c
    ag_uc = torch.matmul(base_ag.view(batch, length, 1), base_uc.view(batch, 1, length))
    
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    gc = torch.matmul(base_g.view(batch, length, 1), base_c.view(batch, 1, length))
    gu = torch.matmul(base_g.view(batch, length, 1), base_u.view(batch, 1, length))
    m1 = (au + gc + gu).to(device)
    m1 = m1 + torch.transpose(m1, 1, 2)
    
    m3 = 1 - diags([1] * 7, [-3, -2, -1, 0, 1, 2, 3], shape=(u.shape[-2], u.shape[-1])).toarray()
    m3 = torch.from_numpy(m3).to(device)
    
    return m1 * m3
  
def postprocess_new(y_sink, y_pred, s, o, coeff_act=10, delta=0.01, mod='original'):
    condition = (y_pred - s) > 0
    conditioned_matrix = condition.double()
    conditioned_matrix.required_grad = False

    y_sink_activated = torch.sigmoid(coeff_act*(y_sink - o))

    if mod == 'original':
        output = (y_sink * conditioned_matrix > o).double()
    elif mod == 'sigmoid':
        output = y_sink_activated * conditioned_matrix
    elif mod == 'softmax':
        constant_column = delta*torch.ones(1,y_sink.size(-2),1).to(y_sink.device)
        y_sink_concat = torch.cat((y_sink,constant_column), dim=2)
        y_sink_softmax = torch.softmax(coeff_act*y_sink_concat, dim=-1)[:,:,:-1]
        output = ((y_sink_softmax + torch.transpose(y_sink_softmax,-1,-2))/2)
    else:
        4/0

    return output


def inference_new(y_sink, y_pred, s, o, coeff_act=10, delta=0.01, mod='original'):
    condition = (y_pred - s) > 0
    conditioned_matrix = condition.double()
    conditioned_matrix.required_grad = False

    y_sink_activated = torch.sigmoid(coeff_act*(y_sink - o))

    if mod == 'original':
        output = (y_sink * conditioned_matrix > o).double()
    elif mod == 'sigmoid':
        output = (y_sink_activated * conditioned_matrix > o).double()
    elif mod == 'softmax':
        constant_column = delta*torch.ones(1,y_sink.size(-2),1).to(y_sink.device)
        y_sink_concat = torch.cat((y_sink,constant_column), dim=2)
        y_sink_softmax = torch.softmax(coeff_act*y_sink_concat, dim=-1)[:,:,:-1]
        output = ((y_sink_softmax + torch.transpose(y_sink_softmax,-1,-2))/2 > o).double()
    else:
        4/0

    return output


def constraint_matrix_proposed(u, x, s, process_device):
    device = torch.device('cuda:{}'.format(process_device))

    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    base_ag = base_a + base_g
    base_uc = base_u + base_c
    ag_uc = torch.matmul(base_ag.view(batch, length, 1), base_uc.view(batch, 1, length))
    
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    gc = torch.matmul(base_g.view(batch, length, 1), base_c.view(batch, 1, length))
    gu = torch.matmul(base_g.view(batch, length, 1), base_u.view(batch, 1, length))
    m1 = au + gc + gu
    
    m2 = (u>s)
    
    m3 = 1 - diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], shape=(u.shape[-2], u.shape[-1])).toarray()
    m3 = torch.Tensor(m3).to(device)
    
    return m1*m2*m3

def postprocess_proposed_us(u, x, s=math.log(9.0), process_device=0, rho=0):
  m = constraint_matrix_proposed(u, x, s, process_device)
  
  cur_u = (u-s) * m
  
  batch_size = x.shape[0]
  
  a = torch.zeros_like(u)
  
  for batch in range(batch_size):
      row_ind, col_ind = linear_sum_assignment(- cur_u[batch].cpu())
      a[batch, row_ind, col_ind] = 1
  
  b = a * m
  b = b + torch.transpose(b, -1, -2)              
  
  result = b.cpu().numpy()
  
  result[result>0] = 1
  
  # print(result)
  
  re = torch.tensor(result)
  
  device = torch.device('cuda:{}'.format(process_device))
  
  return re.to(device).type(torch.cuda.FloatTensor), a

data_type = 'ncRNAs'
model_seed = 0
data_split_seed = 0
device_num = 1

batch_n = 1
epoch_n = 150
set_gamma = 1.5
set_rho = set_gamma + 0.1
set_L1 = 1

# Hyperparameters for Sinkhorn algorithm
max_iter = 10
sinkhorn_threshold = 0.05
sinkhorn_lambda = 1.5
postprocess_threshold = set_gamma
postprocess_output_threshold = 0.2

# Hyperparameters for the combined loss function
c_stage1 = 1
c_stage2 = 1

parser = argparse.ArgumentParser()
# General settings
parser.add_argument('--data_type', type=str, default='ncRNAs', help="Type of data (default: 'ncRNAs')")
parser.add_argument('--model_seed', type=int, default=0, help="Seed for the model initialization (default: 0)")
parser.add_argument('--data_split_seed', type=int, default=0, help="Seed for splitting the data (default: 0)")
parser.add_argument('--device_num', type=int, default=1, help="GPU device number to use (default: 1)")

# Training settings
parser.add_argument('--batch_n', type=int, default=1, help="Batch size (default: 1)")
parser.add_argument('--epoch_n', type=int, default=200, help="Number of epochs (default: 150)")
parser.add_argument('--set_gamma', type=float, default=1.5, help="Gamma parameter (default: 1.5)")
parser.add_argument('--set_rho', type=float, default=1.6, help="Rho parameter (default: set_gamma + 0.1)")
parser.add_argument('--set_L1', type=int, default=1, help="L1 regularization parameter (default: 1)")

# Sinkhorn algorithm hyperparameters
parser.add_argument('--max_iter', type=int, default=10, help="Maximum number of iterations for Sinkhorn algorithm (default: 10)")
parser.add_argument('--sinkhorn_threshold', type=float, default=0.05, help="Threshold for Sinkhorn algorithm (default: 0.05)")
parser.add_argument('--sinkhorn_lambda', type=float, default=1.5, help="Lambda parameter for Sinkhorn algorithm (default: 1.5)")
parser.add_argument('--postprocess_threshold', type=float, default=1.5, help="Threshold for postprocessing (default: set_gamma)")
parser.add_argument('--postprocess_output_threshold', type=float, default=0.2, help="Output threshold for postprocessing (default: 0.2)")

# Combined loss function hyperparameters
parser.add_argument('--loss_type', type=str, default='combined', help="combined/solo")
parser.add_argument('--act_type', type=str, default='sigmoid', help="original/sigmoid/softmax")
parser.add_argument('--c_stage1', type=float, default=1.0, help="Weight for stage 1 in combined loss function (default: 1)")
parser.add_argument('--c_stage2', type=float, default=1.0, help="Weight for stage 2 in combined loss function (default: 1)")
parser.add_argument('--act_ampt', type=float, default=10.0, help="Amplitude inside the activation function")
parser.add_argument('--act_delta', type=float, default=0.01, help="Magnitude of the auxiliary column in the softmax activation")
parser.add_argument('--continue_training', type=bool, default=False, help="continue training using a pretrained model")
parser.add_argument('--model_path', type=str, default='', help="path of the pretrained model")

args = parser.parse_args()
data_split_seed = args.data_split_seed

if device_num != args.device_num:
    device_num = args.device_num
if c_stage1 != args.c_stage1:
    c_stage1 = args.c_stage1
if c_stage2 != args.c_stage2:
    c_stage2 = args.c_stage2
if epoch_n != args.epoch_n:
    epoch_n = args.epoch_n
continue_training = args.continue_training
model_path = args.model_path
if continue_training:
    if not args.model_path:
        4/0

loss_type = args.loss_type
act_type = args.act_type
act_ampt = args.act_ampt
act_delta = args.act_delta

# model_load_path = '/media/ksj/ml_models/REDfold/{}/{}_seed_{}/redfold_{}_model_seed_{}_50th_epoch.pt'.format(data_type,
#                                                                                                               'split' if data_type == 'ncRNAs' else 'model',
#                                                                                                               data_split_seed if data_type == 'ncRNAs' else model_seed,
#                                                                                                               data_type,
#                                                                                                               '0_data_split_seed_{}'.format(data_split_seed) if data_type == 'ncRNAs' else model_seed
#                                                                                                               )

train_data_path = ''
val_data_path = ''
if data_type == 'RNAStrAlign':
    train_data_path = '/media/ksj/final_pickle/RNAStrAlign/REDfold/redfold_redundant_train.pickle'
    val_data_path = '/media/ksj/final_pickle/RNAStrAlign/REDfold/redfold_redundant_val.pickle'
elif data_type == 'ncRNAs':
    train_data_path = '/media/ksj/final_pickle/ncRNAs/REDfold/split_seed_{}/redfold_train_split_seed_{}.pickle'.format(data_split_seed, data_split_seed)
    val_data_path = '/media/ksj/final_pickle/ncRNAs/REDfold/split_seed_{}/redfold_val_split_seed_{}.pickle'.format(data_split_seed, data_split_seed)

if loss_type == 'combined':
    model_save_path = '/media/ksj/so_train/{}/no_pretrained/softmax_postprocess_combinedloss_sigmoid/{}_seed_{}'.format(data_type, 'split' if data_type == 'ncRNAs' else 'model', data_split_seed if data_type == 'ncRNAs' else model_seed)
    model_save_path = '/media/ksj/RNASecondaryStructure/REDfold/train/model_ckpt/{}/{}/{}/{}/c1_{}_c2_{}'.format(data_type,
                                                                                                        'pretrained' if continue_training  else 'no_pretrained',
                                                                                                        'combined' if loss_type=='combined' else 'solo',
                                                                                                        act_type,
                                                                                                        c_stage1,c_stage2
                                                                                                     )
else: 
    model_save_path = '/media/ksj/RNASecondaryStructure/REDfold/train/model_ckpt/{}/{}/{}/{}/act_ampt_{}_act_delta_{}'.format(data_type,
                                                                                                        'pretrained' if continue_training  else 'no_pretrained',
                                                                                                        'combined' if loss_type=='combined' else 'solo',
                                                                                                        act_type,
                                                                                                        act_ampt,act_delta
                                                                                                        )

RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'

device = torch.device('cuda:{}'.format(device_num))

print('model_seed: {}\n'.format(model_seed))
print('gpu_num: {}\n'.format(device_num))
print('data_split_seed: {}\n'.format(data_split_seed))

torch.manual_seed(model_seed)
torch.cuda.manual_seed(model_seed)
torch.cuda.manual_seed_all(model_seed) # gpu 1개 이상일 때 

# 넘파이 랜덤시드 고정
np.random.seed(model_seed)

#CuDNN 랜덤시드 고정
cudnn.benchmark = False
cudnn.deterministic = True # 연산 처리 속도가 줄어들어서 연구 후반기에 사용하자

# 파이썬 랜덤시드 고정
random.seed(model_seed)

Use_gpu = torch.cuda.is_available()

#positive set balance weight for loss function
loss_weight = torch.Tensor([300]).to(device)

print('train data loading...')

train_data = RNASSDataGenerator(train_data_path, 720)
train_len = len(train_data)
train_set = Dataset_FCN(train_data)

dataloader_train = DataLoader(dataset=train_set, batch_size=batch_n, shuffle=1, num_workers=12)

print('valid data loading...')

valid_data = RNASSDataGenerator(val_data_path, 720)
valid_len = len(valid_data)
valid_set = Dataset_FCN(valid_data)

dataloader_valid = DataLoader(dataset=valid_set, batch_size=batch_n, shuffle=1, num_workers=12)

#- Network
model = FCDenseNet(in_channels=146, out_channels=1,
                   initial_num_features=16,
                   dropout=0,
                   down_dense_growth_rates=(4, 8, 16, 32),
                   down_dense_bottleneck_ratios=None,
                   down_dense_num_layers=(4, 4, 4, 4),
                   down_transition_compression_factors=1.0,
                   middle_dense_growth_rate=32,
                   middle_dense_bottleneck=None,
                   middle_dense_num_layers=8,
                   up_dense_growth_rates=(64, 32, 16, 8),
                   up_dense_bottleneck_ratios=None,
                   up_dense_num_layers=(4, 4, 4, 4))

# 모델 파라미터를 더블로 변경
model = model.double()

if continue_training:
    mod_state= torch.load(model_path, map_location=device)
    model.load_state_dict(mod_state['state_dict'])

# Model on GPU
if Use_gpu:
    model = model.to(device)

loss_f = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)

# sinkhorn 돌릴때
loss_s = torch.nn.BCELoss(reduction='none')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
  
loss_record_file_name = ''

if data_type == 'RNAStrAlign':
    loss_record_file_name = '/loss_model_seed_{}.txt'.format(model_seed)
elif data_type == 'ncRNAs':
    loss_record_file_name = '/loss_model_seed_{}_data_split_seed_{}.txt'.format(model_seed, data_split_seed)
    
with open(model_save_path + loss_record_file_name, 'w') as fr:
    fr.write('train_loss,val_loss\n')
  
for epoch in range(epoch_n):
    running_loss_train = 0
    running_loss_val = 0
    
    print(f"-"*10)
    print(f"Epoch {epoch}/{epoch_n}")
    print(f"-"*10)
    print("Phase train...")

    model.train()
        
    print(f"Train data:{len(dataloader_train)}")

    idx = 0
    for index, [x1, y1, L1, seq_hot, seq_name] in enumerate(tqdm(dataloader_train, desc='training...', ascii=True)):
       # Data on GPU
        if Use_gpu:
            x1 = x1.to(device).double()
            y1 = y1.to(device).double()

        [x1, y1] = Variable(x1), Variable(y1)

        y_pred = model(x1)
        y_sink = sinkhorn(y_pred, seq_hot, max_iter, sinkhorn_threshold, sinkhorn_lambda, 1e-7, device_num)
        
        np.set_printoptions(threshold=np.inf)
        # print(y_sink.cpu().detach().numpy())

        # Mask Matrix
        mask1 = torch.zeros_like(y_pred)
        mask1[:, :L1, :L1] = 1

        y_mask = y_sink * mask1
        
        y_mask_new = postprocess_new(y_mask, y_pred, postprocess_threshold, postprocess_output_threshold, coeff_act=act_ampt, delta=act_delta, mod=act_type)

        optimizer.zero_grad()
        
        weight = torch.ones_like(y1).to(device)
        weight[y1 == 1] = 300
        
        if loss_type == 'combined':
            loss_train_stage1 = loss_f(y_pred * mask1, y1).to(device)
            loss_train_stage2 = loss_s(y_mask_new, y1).to(device) # postprocess
            
            loss_train_stage2_weighted = torch.mean(weight.to(device) * loss_train_stage2.to(device)).to(device)
            loss_train_weighted = c_stage1 * loss_train_stage1 + c_stage2 * loss_train_stage2_weighted
        else:
            loss_train_stage2 = loss_s(y_mask_new, y1).to(device)
            loss_train_stage2_weighted = torch.mean(weight.to(device) * loss_train_stage2.to(device)).to(device)
            loss_train_weighted = loss_train_stage2_weighted
        
        loss_train_weighted.backward()
        optimizer.step()

        running_loss_train += loss_train_weighted.item()

    epoch_loss_train = running_loss_train * batch_n / train_len
    
    print(f"Epoch Loss Train:{epoch_loss_train:.4f}")   

    # 모델 저장
    model_name = ''
    if data_type == 'RNAStrAlign':
        model_name = '/redfold_{}_model_seed_{}_{}th_epoch.pt'.format(data_type, model_seed, epoch + 1)
    elif data_type == 'ncRNAs':
        model_name = '/redfold_{}_model_seed_{}_data_split_seed_{}_{}th_epoch.pt'.format(data_type, model_seed, data_split_seed, epoch + 1)
    
    mod_state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
    torch.save(mod_state, model_save_path + model_name)
        
    print(f"Validation data:{len(dataloader_valid)}")

    model.eval()
    idx = 0
    with open(os.path.join(model_save_path, 'redfold_metric_{}th+epoch.csv'.format(epoch)), 'w') as f:
        fw = csv.writer(f)
        fw.writerow(['seq_name','seq_len','ppv','sen','f1','acc'])
        
    for index, [x1, y1, L1, seq_hot, seq_name] in enumerate(tqdm(dataloader_valid, desc='validating...', ascii=True)):
        # Data on GPU
        if Use_gpu:
            x1 = x1.to(device).double()
            y1 = y1.to(device).double()

        [x1, y1] = Variable(x1), Variable(y1)
        
        with torch.no_grad():
            y_pred = model(x1)
            y_sink = sinkhorn(y_pred, seq_hot, max_iter, sinkhorn_threshold, sinkhorn_lambda, eps=1e-8, device_num=device_num)
        
        mask1 = torch.zeros_like(y_pred)
        mask1[:, :L1, :L1] = 1

        y_mask = y_sink * mask1
        
        # post-processing without learning train
        seq_hot = seq_hot.to(device).double()
        
        y_mask_new = inference_new(y_mask, y_pred, postprocess_threshold, postprocess_output_threshold, coeff_act=act_ampt, delta=act_delta, mod=act_type)
        
        optimizer.zero_grad()
        
        weight = torch.ones_like(y1).to(device)
        weight[y1 == 1] = 300
        
        if loss_type == 'combined':
            loss_val_stage1 = loss_f(y_pred * mask1, y1).to(device)
            loss_val_stage2 = loss_s(y_mask_new, y1).to(device) # postprocess
            
            loss_val_stage2_weighted = torch.mean(weight.to(device) * loss_val_stage2.to(device)).to(device)
            loss_val_weighted = c_stage1 * loss_train_stage1 + c_stage2 * loss_val_stage2_weighted
        else:
            loss_val_stage2 = loss_s(y_mask_new, y1).to(device)
            loss_val_stage2_weighted = torch.mean(weight.to(device) * loss_val_stage2.to(device)).to(device)
            loss_val_weighted = loss_val_stage2_weighted
        
        running_loss_val += loss_val_weighted.item()
        
        positive_predictive_value, sensitivity, f1_score, accuracy = evaluate_exact_a(y_mask_new.cpu(), y1.cpu())
        
        with open(os.path.join(model_save_path, 'redfold_metric_{}th+epoch.csv'.format(epoch)), 'a') as f:
            fw = csv.writer(f)
            fw.writerow([seq_name, int(L1[0]),
                        positive_predictive_value.item(), sensitivity.item(),
                        f1_score.item(), accuracy.item()])
    
    epoch_loss_val = running_loss_val * batch_n / valid_len
    
    with open(model_save_path + loss_record_file_name, 'a') as fr:
        fr.write('{},{}\n'.format(epoch_loss_train, running_loss_val))
