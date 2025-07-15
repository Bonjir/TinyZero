
# python ./tests/test_dataset_2.py
import numpy as np
import pandas as pd

train_path = '/NAS/chenfeng/dataset/countdown/train.parquet'
test_path = '/NAS/chenfeng/dataset/countdown/test.parquet'

def load_data(path):
    return pd.read_parquet(path, engine='pyarrow')


train_data = load_data(train_path)
print(train_data.head())

test_data = load_data(test_path)
print(test_data.head())

print(f'------------------------------------------------------')
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

print(f"Train data columns: {train_data.columns.tolist()}")
print(f"Test data columns: {test_data.columns.tolist()}")

print(f'------------------------------------------------------')

print(f"Train data sample: {train_data.iloc[0]}")

print(f'train_data_describe:\n{train_data.describe()}')

for col in train_data.columns:
    try:
        print(f'Unique values in "{col}" column ({train_data[col].nunique()}): {train_data[col].unique()}')
    except Exception as e:
        print(f'Error processing column "{col}": {e}')
        
print(f'------------------------------------------------------')

for row in range(5):
    print(f'prompt of sample {row}: {train_data.loc[row, "prompt"]}')
    
flag = 0
for row in range(len(train_data)):
# for row in range(5):
    prompt_this = train_data.loc[row, 'prompt']
    if len(prompt_this) != 1:
        print(f'Error in sample {row}: prompt is not a list or len={len(prompt_this)}, type={type(prompt_this)}')
        flag += 1
        continue
    prompt_str = prompt_this[0]['content']
    if not isinstance(prompt_str, str):
        print(f'Error in sample {row}: prompt is not a string')
        flag += 1
        continue
    if r'(+, -, *, /)' not in prompt_str:
        print(f'Error in sample {row}: {prompt_str}')
        flag += 1   
        
if flag == 0:
    print(f'All prompts contain "(+, -, *, /)"')
else:
    print(f'Found {flag} samples without "(+, -, *, /)" in prompt')