import numpy as np
import os
from tqdm import tqdm
import argparse
import subprocess

# for SPOT-RNA
from SPOT_RNA.utils.utils import create_tfr_files
from SPOT_RNA.utils.FastaMLtoSL import FastaMLtoSL
import tensorflow as tf

# for E2Efold & REDfold
import torch
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
import collections
import dill as cPickle
from random import shuffle
import pandas as pd
import subprocess
from Bio import SeqIO
from itertools import product

from E2Efold.model.models import ContactAttention_simple_fix_PE, Lag_PP_mixed, RNA_SS_e2e
from E2Efold.utils.utils import *
from E2Efold.data_generator.data_generator import RNASSDataGenerator as RNASSDataGeneratorE2E, Dataset
from E2Efold.utils.postprocess import  postprocess_proposed as postprocess_proposed_e2e

from REDfold.utils.utils import *
from REDfold.data_generator.data_generator import RNASSDataGenerator as RNASSDataGeneratorRED
from REDfold.data_generator.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from REDfold.model.models import FCDenseNet
from REDfold.utils.postprocess import postprocess_proposed as postprocess_proposed_red

def generate_visible_device(n):
  sequence = ','.join(map(str, range(n + 1)))
  return sequence

def E2E_pickle_make(seq_name, sequence, output_path):
  RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
  label_dict = {
    '.': np.array([1,0,0]), 
    '(': np.array([0,1,0]), 
    ')': np.array([0,0,1])
  }
  seq_dict = {
      'A':np.array([1,0,0,0]),
      'U':np.array([0,1,0,0]),
      'C':np.array([0,0,1,0]),
      'G':np.array([0,0,0,1]),
      'N':np.array([0,0,0,0])
  }

  def seq_encoding(string):
      str_list = list(string)
      encoding = list(map(lambda x: seq_dict[x], str_list))
      # need to stack
      return np.stack(encoding, axis=0)

  def stru_encoding(string):
      str_list = list(string)
      encoding = list(map(lambda x: label_dict[x], str_list))
      # need to stack
      return np.stack(encoding, axis=0)

  def padding(data_array, maxlen):
      a, b = data_array.shape
      return np.pad(data_array, ((0,maxlen-a),(0,0)), 'constant')
  
  length_limit = 600
  
  structure_list = list()
  seq_list = list()
  
  seq_len_list = list()
  file_list = list()
  pairs_list = list()
  
  file_list.append(seq_name)
  seq_len_list.append(len(sequence))
  pairs_list.append([])
  
  structure_list.append('.'*len(sequence))
  seq_list.append(sequence.upper())
  
  seq_encoding_list = list(map(seq_encoding, seq_list))
  stru_encoding_list = list(map(stru_encoding, structure_list))
  
  seq_encoding_list_padded = list(map(lambda x: padding(x, length_limit), 
    seq_encoding_list))
  stru_encoding_list_padded = list(map(lambda x: padding(x, length_limit), 
    stru_encoding_list))
  
  
  RNA_SS_data_list = list()
  for i in range(1):
      RNA_SS_data_list.append(RNA_SS_data(seq=seq_encoding_list_padded[i],
          ss_label=stru_encoding_list_padded[i], 
          length=seq_len_list[i], name=file_list[i], pairs=pairs_list[i]))
      
  with open(os.path.join(output_path, seq_name+'.pickle') , 'wb') as f:
    cPickle.dump(RNA_SS_data_list, f)
    
def RED_pickle_make(seq_name, sequence, output_path):
  RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
  
  BASE1 = 'AUCG'
  pair_set= {'AU','UA','CG','GC','GU','UG'}

  global npBASE1
  global dcBASE2
  
  npBASE1= np.array([b1 for b1 in BASE1])
  npBASE2= np.array(["".join(b2) for b2 in product(npBASE1,npBASE1)])
  dcBASE2= {}
  for [a,b] in enumerate(npBASE2):
    dcBASE2[b]= a

  def one_hot(seq1):
      RNA_seq= seq1

      feat= np.concatenate([[(npBASE1 == base.upper()).astype(int)] 
            if str(base).upper() in BASE1 else np.array([[0] * len(BASE1)]) for base in RNA_seq])

      return feat

  def one_hot_2m(seq1):
      L1= len(seq1)
      feat= np.zeros((L1,16))
      for i in range(0,L1-1):
        Id1= str(seq1[i:i+2]).upper()
        if Id1 in dcBASE2:
          feat[i,dcBASE2[Id1]]= 1
      #Circle Back 2mer
      Id1= str(seq1[-1]+seq1[0]).upper()
      feat[L1-1,dcBASE2[Id1]]= 1

      return feat

  def get_cut_len(data_len,set_len):
      L= data_len
      if L<= set_len:
          L= set_len
      else:
          L= (((L - 1) // 16) + 1) * 16
      return L

  def pair2map(pairs, seq_len):
    pmap= np.zeros([seq_len, seq_len])
    for pair in pairs:
      pmap[pair[0], pair[1]] = 1
    return pmap
  
  all_files_list = []
  
  one_hot_matrix= one_hot(sequence.upper())
  one_hot_mat2= one_hot_2m(sequence.upper())
  
  pair_dict_all_list = []
  
  seq_name = seq_name
  seq_len = len(sequence)
  
  # pair_dict_all = dict()
  
  # ss_label = np.zeros((seq_len,3),dtype=int)
  
  L= get_cut_len(seq_len,80)
  
  ##-Trans seq to seq_length
  one_hot_matrix_LM= np.zeros((L,4))
  one_hot_matrix_LM[:seq_len,]= one_hot_matrix
  # ss_label_L= np.zeros((L,3),dtype=int)

  one_hot_mat2_LM= np.zeros((L,16))
  one_hot_mat2_LM[:seq_len,]= one_hot_mat2
  
  data_seq1= one_hot_matrix_LM
  data_seq2= one_hot_mat2_LM

  ##-Seq_onehot
  seq_hot= one_hot_matrix_LM[:L,:]
  data_pair= pair2map(pair_dict_all_list,L)
  
  sample_tmp= RNA_SS_data(name=seq_name, length=seq_len, seq_hot=seq_hot, data_pair=data_pair, data_seq1= data_seq1, data_seq2=data_seq2)      
  all_files_list.append(sample_tmp)
  
  with open(os.path.join(output_path, seq_name+'.pickle'), 'wb') as f:
    cPickle.dump(all_files_list, f)
    
def SPOT_fasta_make(seq_name, sequence, output_path):
  lines = ['>'+seq_name+'\n', sequence.upper()+'\n']
  with open(os.path.join(output_path, seq_name+'.fasta'), 'w') as f:
    f.writelines(lines)
    
def exac_E2E(pickle_path, num_device=1, device_num = 0):
  # nc seed 2 epoch 20 s 4
  model_path = './E2Efold/model/e2efold_model_new.pt'
  d = 10
  BATCH_SIZE = 1
  pp_steps = 20
  k = 1
  s = 4
  
  # 여기 개수에 따라 바꿔줘야 함
  os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_device)

  device = torch.device('cuda:{}'.format(device_num))
  
  RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
  
  test_data = RNASSDataGeneratorE2E(pickle_path)

  seq_len = test_data.data_y.shape[-2]

  params = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 6,
            'drop_last': True}
  
  test_set = Dataset(test_data)
  test_generator = data.DataLoader(test_set, **params)

  contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len).to(device)
  lag_pp_net = Lag_PP_mixed(pp_steps, k, device=device_num).to(device)
  rna_ss_e2e = RNA_SS_e2e(contact_net.to(device), lag_pp_net.to(device)).to(device)
  rna_ss_e2e.load_state_dict(torch.load(model_path, map_location = device))
  rna_ss_e2e.to(device)

  contact_net = rna_ss_e2e.model_att
  lag_pp_net = rna_ss_e2e.model_pp

  contact_net.eval()
  lag_pp_net.eval()
  rna_ss_e2e.eval()
  
  for index, [contacts, seq_embeddings, matrix_reps, seq_lens, data_name] in enumerate(tqdm(test_generator, desc='test data loading...', ascii=True)):
    
    contacts_batch = torch.Tensor(contacts.float()).to(device)
    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
    # matrix_reps_batch = torch.unsqueeze(
    #     torch.Tensor(matrix_reps.float()).to(device), -1)

    state_pad = torch.zeros(contacts.shape).to(device)

    PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
    
    with torch.no_grad():
      pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, seq_embedding_batch, state_pad)

      u_no_train_2 = postprocess_proposed_e2e(pred_contacts[:,:seq_lens, :seq_lens], seq_embedding_batch[:,:seq_lens,:seq_lens],s=s, process_device=device_num)
      
      return(u_no_train_2[0].cpu().numpy())


def exac_RED(pickle_path, num_device=1,device_num = 0):
  # nc seed 2 epoch 124 s 0
  model_path = './REDfold/model/redfold_model_new.pt'

  # 나중에 바꾸기
  use_device_num = device_num
  batch_n = 1
  set_gamma = 0
  set_rho= set_gamma+0.1
  set_L1= 1
  
  RNA_SS_data = collections.namedtuple('RNA_SS_data','name length seq_hot data_pair data_seq1 data_seq2')
  
  os.environ["CUDA_VISIBLE_DEVICES"] = generate_visible_device(num_device)

  device = torch.device('cuda:{}'.format(use_device_num))

  Use_gpu= torch.cuda.is_available()

  print('test data loading...')

  test_data= RNASSDataGeneratorRED(pickle_path,720)
  test_set= Dataset_FCN(test_data)

  dataloader_test= DataLoader(dataset=test_set, batch_size=batch_n, shuffle=1, num_workers=12)

  #- Network
  model= FCDenseNet(in_channels=146,out_channels=1,
                  initial_num_features=16,
                  dropout=0,

                  down_dense_growth_rates=(4,8,16,32),
                  down_dense_bottleneck_ratios=None,
                  down_dense_num_layers=(4,4,4,4),
                  down_transition_compression_factors=1.0,

                  middle_dense_growth_rate=32,
                  middle_dense_bottleneck=None,
                  middle_dense_num_layers=8,

                  up_dense_growth_rates=(64,32,16,8),
                  up_dense_bottleneck_ratios=None,
                  up_dense_num_layers=(4,4,4,4))

  optimizer= torch.optim.Adam(model.parameters())

  # Model on GPU
  if Use_gpu:
      model= model.to(device)

  mod_state= torch.load(model_path, map_location=device)
  model.load_state_dict(mod_state)

  model.eval()
  
  for index, [x1, y1, L1, seq_hot,seq_name] in enumerate(tqdm(dataloader_test, desc='testing...', ascii=True)): 
    # Data on GPU
    if Use_gpu:
        x1= x1.to(device).type(torch.cuda.FloatTensor)
        y1= y1.to(device).type(torch.cuda.FloatTensor)

    [x1, y1]= Variable(x1), Variable(y1)
    
    with torch.no_grad():
        y_pred= model(x1)
    
    # post-processing without learning train
    seq_hot=seq_hot.to(device)
    y_mask_proposed = postprocess_proposed_red(y_pred, seq_hot, set_gamma, use_device_num)

    optimizer.zero_grad()
    
    print(y_mask_proposed[0][:L1,:L1].shape)
    
    return(y_mask_proposed[0][:L1,:L1].cpu().numpy())


def exac_SPOT(fasta_path, device_num):
  # tfr_path = os.path.dirname(fasta_path)
  
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  FastaMLtoSL(fasta_path)

  base_path = os.path.dirname(fasta_path)
  input_file = os.path.basename(fasta_path)
  
  if not os.path.exists(os.path.join(base_path, 'input_tfr_files')):
    os.makedirs(os.path.join(base_path, 'input_tfr_files'))
    print('folder make done')

  create_tfr_files(fasta_path, base_path, input_file)
  
  with open(fasta_path) as file:
    input_data = [line.strip() for line in file.read().splitlines() if line.strip()]

  count = int(len(input_data)/2)

  ids = [input_data[2*i].replace(">", "") for i in range(count)]
  sequences = {}
  
  for i,I in enumerate(ids):
    sequences[I] = input_data[2*i+1].replace(" ", "").upper().replace("T", "U")

  os.environ["CUDA_VISIBLE_DEVICES"]= str(device_num)
  #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  NUM_MODELS = 5

  test_loc = [os.path.join(base_path, 'input_tfr_files', input_file+'.tfrecords')]

  outputs = {}
  mask = {}
  def sigmoid(x):
      return 1/(1+np.exp(-np.array(x, dtype=np.float128)))
  
  for MODEL in range(NUM_MODELS):
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement=False
    config.gpu_options.allow_growth = True

    print('\nPredicting for SPOT-RNA model '+str(MODEL))
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(os.path.join('./SPOT_RNA/SPOT-RNA-models', 'model' + str(MODEL) + '.meta'))
        saver.restore(sess,os.path.join('./SPOT_RNA/SPOT-RNA-models', 'model' + str(MODEL)))
        graph = tf.compat.v1.get_default_graph()
        init_test =  graph.get_operation_by_name('make_initializer_2')
        tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')
        name_tensor = graph.get_tensor_by_name('tensors_2/component_0:0')
        RNA_name = graph.get_tensor_by_name('IteratorGetNext:0')
        label_mask = graph.get_tensor_by_name('IteratorGetNext:4')
        sess.run(init_test,feed_dict={name_tensor:test_loc})
        
        pbar = tqdm(total = count)
        while True:
            try:        
                out = sess.run([tmp_out,RNA_name,label_mask],feed_dict={'dropout:0':1})
                out[1] = out[1].decode()
                mask[out[1]] = out[2]
                
                if MODEL == 0:
                    outputs[out[1]] = [sigmoid(out[0])]
                else:
                    outputs[out[1]].append(sigmoid(out[0]))
                #print('RNA name: %s'%(out[1]))
                pbar.update(1)
            except tf.errors.OutOfRangeError:
                break
        pbar.close()
    tf.compat.v1.reset_default_graph()
  RNA_ids = [i for i in list(outputs.keys())]
  ensemble_outputs = {} 
  
  def output_mask(seq, NC=True):
      if NC:
          include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
      else:
          include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
      mask = np.zeros((len(seq), len(seq)))
      for i, I in enumerate(seq):
          for j, J in enumerate(seq):
              if str(I) + str(J) in include_pairs:
                  mask[i, j] = 1
      return mask
  
  for i in RNA_ids:
    ensemble_outputs[i] = np.mean(outputs[i],0)
    
    # print('\n\n\n\n\n shape: {} \n\n\n\n\n'.format(np.array(ensemble_outputs[i]).shape))
    
    ensemble_outputs_post = ensemble_outputs[i]
    label_mask = mask[i]
    seq = sequences[i]
    name = i
    # Threshold = 0.335
    test_output = ensemble_outputs_post
    mask_post = output_mask(seq)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]
    y_pred = np.multiply(y_pred, mask_post)
    
    return(y_pred)

# matrix를 ct 파일로 바꿔주는 함수
def write_ct_file(rna_name, sequence, adjacency_matrix, output_file):
    """
    Create a .ct file from RNA name, sequence, and adjacency matrix with fixed-width formatting and right alignment.
    
    :param rna_name: Name of the RNA molecule
    :param sequence: RNA sequence
    :param adjacency_matrix: n*n adjacency matrix representing secondary structure
    :param output_file: Path to save the .ct file
    """
    n = len(sequence)
    if len(adjacency_matrix) != n or any(len(row) != n for row in adjacency_matrix):
        raise ValueError("The adjacency matrix dimensions must match the sequence length.")
    
    # Generate pair information from adjacency matrix
    paired_positions = [0] * n
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                paired_positions[i] = j + 1  # Convert to 1-based index

    # Calculate field width based on max digit count
    max_index_width = len(str(n))
    field_width = max_index_width + 1

    # Format string for fixed-width columns with right alignment
    fmt = f"{{:>{field_width}}} {{:<2}} {{:>{field_width}}} {{:>{field_width}}} {{:>{field_width}}} {{:>{field_width}}}\n"
    
    # Write to .ct file
    with open(output_file, "w") as f:
        f.write(f"{n} {rna_name}\n")
        for i in range(n):
            f.write(fmt.format(
                i + 1,  # 1-based index
                sequence[i],  # Nucleotide
                i if i > 0 else 0,  # Previous index (0 if first nucleotide)
                i + 2 if i + 1 < n else 0,  # Next index (0 if last nucleotide)
                paired_positions[i],  # Paired index (0 if unpaired)
                i + 1  # 1-based index (repeated)
            ))
    
    print(f".ct file written to {output_file}")
    
# dot_bracker functions
def listtuple_to_string(list, L):
    strlist = ["."] * L
    for (i,j) in list:
        strlist[i] = "("
        strlist[j] = ")"

    return "".join(strlist)  

def print_pkfreeline(entry, L):

    num_entry = entry.shape[0]

    i_prev,j_prev = 0,num_entry
    list_pkfree = []
    list_remaining = []
    list_check = []
    i_prev,j_prev = 0,0

    for k in range(num_entry):
        i,j = entry[k]

        if(k==0):
            list_pkfree.append((i,j))
            list_check.append((i,j))
        
        else:
            if any((i<j_check and j_check<j) for (i_check,j_check) in list_check):
                list_remaining.append((i,j))
            else:
                list_pkfree.append((i,j))
                if j_prev < j:
                    list_check.append((i,j))
                    if i_prev + 1 < j_prev:
                        list_check.append((i_prev,j_prev))

                if j_prev > j+1:
                    list_check.append((i_prev,j_prev))
        

        i_prev, j_prev = i, j


    rnastr =  listtuple_to_string(list_pkfree, L)

    return rnastr,np.array(list_remaining)


def al2npy(algorithm:str, uuid:str, seq_name:str, sequence:str, base_path:str = './', num_of_gpus:int = 1, gpu_num:int = 0):
  valid_algorithms = ['e2efold', 'redfold', 'spot-rna']
  
  if algorithm not in valid_algorithms:
    raise ValueError(
       f"Invalid algorithm '{algorithm}'. "
      f"Valid options are: {', '.join(valid_algorithms)}."
    )

  tmp_path = os.path.join(base_path,uuid)

  if not os.path.exists(tmp_path):
      os.makedirs(tmp_path)
  else:
    # 이미 있다는 얘기이므로 폴더 안에서 꺼내 쓰면 댐
    print('hi')

  if algorithm == 'spot-rna':
    SPOT_fasta_make(seq_name,sequence,tmp_path)
    spot_npy = exac_SPOT(os.path.join( tmp_path, seq_name+'.fasta'),gpu_num)
    np.save(os.path.join( tmp_path, seq_name+'.npy'), spot_npy)
    write_ct_file(seq_name, sequence, spot_npy, os.path.join( tmp_path, 'rna.ct'))
    subprocess.run(args=['java', '-cp', 'VARNAv3-93.jar', 'fr.orsay.lri.varna.applications.VARNAcmd', 
                         '-i', os.path.join( tmp_path, 'rna.ct'), '-o', 
                         os.path.join( tmp_path, 'image.png'), '-resolution', '10.0'], 
                   cwd="/workspace/nar_web_rna/varna/")
    # dot bracket
    entry = np.transpose(np.nonzero(np.triu(spot_npy)))
    L = spot_npy.shape[0]
    list_remaining = entry
    list_str = []
    while True:
      rnastr,list_remaining = print_pkfreeline(list_remaining, L)
      list_str.append(rnastr)
      if not list_remaining.size==0: continue
      break
    return list_str
  elif algorithm == 'e2efold':
    E2E_pickle_make(seq_name,sequence,tmp_path)
    e2e_npy = exac_E2E(os.path.join( tmp_path, seq_name+'.pickle'))
    np.save(os.path.join( tmp_path, seq_name+'.npy'), e2e_npy)
    write_ct_file(seq_name, sequence, e2e_npy, os.path.join( tmp_path, 'rna.ct'))
    subprocess.run(args=['java', '-cp', 'VARNAv3-93.jar', 'fr.orsay.lri.varna.applications.VARNAcmd', 
                         '-i', os.path.join( tmp_path, 'rna.ct'), '-o', 
                         os.path.join( tmp_path, 'image.png'), '-resolution', '10.0'], 
                   cwd="/workspace/nar_web_rna/varna/")
    # dot bracket
    entry = np.transpose(np.nonzero(np.triu(e2e_npy)))
    L = e2e_npy.shape[0]
    list_remaining = entry
    list_str = []
    while True:
      rnastr,list_remaining = print_pkfreeline(list_remaining, L)
      list_str.append(rnastr)
      if not list_remaining.size==0: continue
      break
    return list_str
  elif algorithm == 'redfold':
    RED_pickle_make(seq_name,sequence,tmp_path)
    red_npy = exac_RED(os.path.join( tmp_path, seq_name+'.pickle'))
    np.save(os.path.join( tmp_path, seq_name+'.npy'), red_npy)
    write_ct_file(seq_name, sequence, red_npy, os.path.join( tmp_path, 'rna.ct'))
    subprocess.run(args=['java', '-cp', 'VARNAv3-93.jar', 'fr.orsay.lri.varna.applications.VARNAcmd', 
                         '-i', os.path.join( tmp_path, 'rna.ct'), '-o', 
                         os.path.join( tmp_path, 'image.png'), '-resolution', '10.0'], 
                   cwd="/workspace/nar_web_rna/varna/")
    # dot bracket
    entry = np.transpose(np.nonzero(np.triu(red_npy)))
    L = red_npy.shape[0]
    list_remaining = entry
    list_str = []
    while True:
      rnastr,list_remaining = print_pkfreeline(list_remaining, L)
      list_str.append(rnastr)
      if not list_remaining.size==0: continue
      break
    return list_str
  else:
    raise ValueError(
       f"Invalid algorithm '{algorithm}'. "
      f"Valid options are: {', '.join(valid_algorithms)}."
    )