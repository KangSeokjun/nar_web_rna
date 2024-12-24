from nar_algorithm2npy import al2npy
import os
import json

dbn = al2npy(algorithm='spot-rna', uuid='test', seq_name='seq_name', sequence='AACUGCGCCGUGUCGGGAAGUGGAAACACUCUCGGAGCUUGCAAUUAGGCUUGAACAACACGUAGCGCAGUUUUAAAAAAAAAAAA', base_path='./datas')
# output_meta = os.path.join('/data', 'test', 'meta.json')

# with open(output_meta, 'w') as json_file:
#   json_dct = {"sequence": "AAAAA", "seq_name": "seq_name", "algorithm": "algorithm"}
#   json.dump(json_dct, json_file)

# with open(output_meta, 'r') as json_file:
#   json_dic = json.load(json_file)
  
# with open(output_meta, 'w') as json_file:
#   json_dic['dot_bracket'] = dbn
#   json.dump(json_dic, json_file)

# os.remove(os.path.join( os.path.join('/data','test'), '.notdone'))

print(dbn)