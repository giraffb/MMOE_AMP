import warnings 
# 忽略所有的 DeprecationWarning 警告
warnings.filterwarnings('ignore')

import os
for feature_dir in ("esm_embedding", "macrel_embedding", "textcnn_convert"):
    os.makedirs(feature_dir, exist_ok=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinTensor
from Bio import SeqIO  # 用于读取 FASTA 文件
import attr
import torch.nn.functional as F
client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"

def add_padding(protein_tensor: ESMProteinTensor, max_length: int, pad_token_id: int = 1) -> ESMProteinTensor:
    """
    Pad the sequence of ESMProteinTensor to max_length with pad_token_id,
    keeping BOS and EOS tokens fixed.
    """
    sequence = protein_tensor.sequence
    current_length = sequence.size(0)

    # 检查是否有BOS/EOS
    BOS_token = sequence[0].item()
    EOS_token = sequence[-1].item()

    # 序列中实际内容（去掉BOS/EOS）长度
    inner_length = current_length - 2
    if current_length > max_length:
        raise ValueError(
            f"Protein length {current_length} exceeds max length {max_length}"
        )

    # 计算需要填充的长度
    right_pad = max_length - 2 - inner_length

    # 提取 inner 序列（去掉 BOS 和 EOS）
    inner_sequence = sequence[1:-1]

    # 在inner序列后面填充
    padded_inner_sequence = F.pad(
        inner_sequence,
        (0, right_pad),
        value=pad_token_id
    )

    # 拼回完整序列：BOS + inner + PAD + EOS
    padded_sequence = torch.cat([
        torch.tensor([BOS_token], device=sequence.device),
        padded_inner_sequence,
        torch.tensor([EOS_token], device=sequence.device)
    ])

    # 创建新的 protein_tensor
    padded_protein_tensor = attr.evolve(
        protein_tensor,
        sequence=padded_sequence
    )

    return padded_protein_tensor

import h5py

#保存HDF5文件
def fasta_to_padded_embeddings(fasta_file, output_prefix):
    """
    从FASTA文件读取蛋白质序列，生成嵌入并填充/截断到固定长度
    
    参数:
    - fasta_file: 输入的FASTA文件路径
    - output_prefix: 输出文件前缀
    """
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    
    all_embeddings = []
    sequence_info = []
    
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        seq_id = record.id
        sequence = str(record.seq)
        #print(f"Processing {i+1}: {seq_id} (Length: {len(sequence)})")
        
        # 创建ESMProtein实例
        protein = ESMProtein(sequence=sequence)
        
        # 编码序列并生成logits
        with torch.no_grad():
            protein_tensor = client.encode(protein)
            protein_tensor_padded = add_padding(protein_tensor, 100+2)
            logits_output = client.logits(
                protein_tensor_padded, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )
        
        # 提取完整嵌入 [1, L+2, 960] -> 去除BOS/EOS后 [L, 960]
        embedding = logits_output.embeddings[0, 1:-1, :].cpu().numpy()
        
        ## 填充或截断嵌入到固定长度
        #padded_embedding = pad_or_truncate(embeddings, MAX_LENGTH)
        all_embeddings.append(embedding)
        
        # 保存序列信息
        sequence_info.append({
            'id': seq_id,
            'original_length': len(sequence),
            'padded_length': embedding.shape[0],
            'sequence': sequence
        })
    
    # 转换为三维NumPy数组 [N, MAX_LENGTH, 960]
    all_embeddings = np.stack(all_embeddings)
    print(all_embeddings.shape)
    
    # ===== 修改开始：保存为HDF5文件 =====
    h5_file = f"{output_prefix}.h5"
    with h5py.File(h5_file, 'w') as hf:
        # 保存嵌入数组
        hf.create_dataset('embeddings', data=all_embeddings)
        
        # 保存序列信息为结构化数组
        dtype = [
            ('id', h5py.string_dtype(encoding='utf-8')),
            ('original_length', int),
            ('padded_length', int),
            ('sequence', h5py.string_dtype(encoding='utf-8'))
        ]
        info_arr = np.array(
            [(info['id'], info['original_length'], info['padded_length'], info['sequence']) 
             for info in sequence_info],
            dtype=dtype
        )
        hf.create_dataset('sequence_info', data=info_arr)
    # ===== 修改结束 =====
    
    print(f"保存完成: {all_embeddings.shape[0]} 个序列的嵌入到 {h5_file} (形状: {all_embeddings.shape})")
    return all_embeddings.shape

## 处理所有数据集
datasets = [
    'AMPBASE_train', 'AMPBASE_val', 'AMPBASE_test', 'nonAMPBASE_train', 'nonAMPBASE_val', 'nonAMPBASE_test'
]


for file in datasets:
    fasta_path = f'../datatset/{file}.fasta'
    output_prefix = f'esm_embedding/{file.split("/")[-1]}'
    shape = fasta_to_padded_embeddings(fasta_path, output_prefix)
    print(f'{file} 处理完成! 嵌入形状: {shape}')

from A_model_def.AMP_features import fasta_features

# 主函数调用
for file in [
    'AMPBASE_train', 'AMPBASE_val', 'AMPBASE_test', 'nonAMPBASE_train', 'nonAMPBASE_val', 'nonAMPBASE_test']:
    fasta_file = '../datatset/'+file+".fasta"   # 输入的 FASTA 文件路径
    output_file = 'macrel_embedding/'+file.split('/')[-1]+".csv"       # 替换为你想要保存的CSV文件路径
    feat = fasta_features(fasta_file)
    feat.iloc[:,2:].to_csv(output_file, index_label='Access')

# 定义序列词汇表(来源ESM)
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

# 创建字符到索引的映射
char_to_index = {char: idx for idx, char in enumerate(SEQUENCE_VOCAB)}

def convert_sequence_to_indices(sequence, max_length=100):
    """
    将蛋白质序列转换为数字索引序列，并填充到固定长度
    
    参数:
        sequence (str): 蛋白质序列字符串
        max_length (int): 目标序列长度
        
    返回:
        list: 包含数字索引的列表，长度固定为max_length
    """
    indices = []
    
    for char in sequence:
        char_upper = char.upper()
        
        # 检查字符是否在词汇表中
        if char_upper in char_to_index:
            indices.append(char_to_index[char_upper])
        else:
            # 使用<unk>索引处理未知字符
            indices.append(char_to_index["<unk>"])
            print('unknown error')
    
    # 序列长度处理
    if len(indices) > max_length:
        indices = indices[:max_length]
        print('len error')
    elif len(indices) < max_length:
        indices += [char_to_index["<pad>"]] * (max_length - len(indices))
    return indices

# ...（前面的SEQUENCE_VOCAB和char_to_index保持不变）...

def process_fasta_file(input_path, output_path):
    """
    处理FASTA文件，添加行名(蛋白ID)和列名(位置索引)
    增强对复杂ID格式的处理能力
    """
    sequence_data = []
    unknown_id_counter = 0  # 用于生成唯一ID
    
    with open(input_path, 'r') as f:
        current_id = None
        current_sequence = ""
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if line.startswith('>'):  # 序列头
                # 保存前一个序列
                if current_id is not None and current_sequence:
                    sequence_data.append((current_id, current_sequence))
                
                # 处理新ID
                id_line = line[1:].strip()  # 去掉'>'和首尾空格
                
                # 生成有效的ID
                if not id_line:  # 空ID行
                    current_id = f"unknown_id_{unknown_id_counter}"
                    unknown_id_counter += 1
                    print(f"警告: 文件{input_path}第{line_num}行发现空ID，已生成唯一ID: {current_id}")
                else:
                    # 清理ID：替换可能引起CSV问题的字符
                    clean_id = id_line.split()[0]  # 取第一个单词
                    clean_id = clean_id.replace(',', '_')  # 替换逗号
                    clean_id = clean_id.replace(';', '_')  # 替换分号
                    clean_id = clean_id.replace('"', '')   # 删除引号
                    current_id = clean_id
                
                current_sequence = ""
            else:
                current_sequence += line
        
        # 添加最后一个序列
        if current_id is not None and current_sequence:
            sequence_data.append((current_id, current_sequence))
    
    # 转换序列为数字索引
    converted_data = []
    for seq_id, seq in sequence_data:
        indices = convert_sequence_to_indices(seq, max_length=100)
        converted_data.append((seq_id, indices))
    
    # 保存为CSV文件（添加行名和列名）
    with open(output_path, 'w') as f:
        # 写入列名: 第一列是"ID"，后面是位置列(pos0, pos1, ..., pos99)
        column_names = ["ID"] + [f"pos{i}" for i in range(100)]
        f.write(",".join(column_names) + "\n")
        
        # 写入数据行
        for seq_id, indices in converted_data:
            # 确保ID不包含逗号等特殊字符
            safe_id = seq_id.replace(',', '_')
            line = safe_id + "," + ",".join(map(str, indices))
            f.write(line + "\n")
    
    print(f"已处理 {len(converted_data)} 条序列，保存到 {output_path}")

# ...（文件处理循环保持不变）...
# 定义要处理的文件列表
file_list = [
    'AMPBASE_train', 'AMPBASE_val', 'AMPBASE_test', 'nonAMPBASE_train', 'nonAMPBASE_val', 'nonAMPBASE_test'
]

# 处理每个文件
for file in file_list:
    fasta_file = f'../datatset/{file}.fasta'
    output_file = f'textcnn_convert/{file.split("/")[-1]}.csv'
    
    print(f"处理文件: {fasta_file}")
    process_fasta_file(fasta_file, output_file)

print("\n所有文件处理完成!")