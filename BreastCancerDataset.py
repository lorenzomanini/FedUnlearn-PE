import pandas as pd
from tqdm import tqdm
import os
import re
import numpy as np
import pickle

from torch.utils.data import Dataset
import torch


class EncodingType:
    ONE_HOT = 'one_hot'
    INT_EMBEDDING = 'int_embedding'
    RAW_INT = 'raw_int'


class DataType:
    INPUT = 'input'
    TARGET = 'target'


def create_breast_cancer_config(dataframe):
    fields = dataframe.columns.tolist()
    config = dict()
    for field in fields:
        print(f"Configuring field: {field}")
        print("Select encoding type:")
        print("1. One-Hot Encoding")
        print("2. Integer Embedding")
        print("3. Raw Integer")
        print("4. Skip field")
        choice = input("Enter choice (1/2/3/4): ")
        if choice == '1':
            encoding = EncodingType.ONE_HOT
        elif choice == '2':
            encoding = EncodingType.INT_EMBEDDING
        elif choice == '3':
            encoding = EncodingType.RAW_INT
        elif choice == '4':
            print(f"Skipping field: {field}")
            continue
        else:
            print("Invalid choice, defaulting to Raw Integer.")
            encoding = EncodingType.RAW_INT

        print("Select data type:")
        print("1. Input")
        print("2. Target")
        choice = input("Enter choice (1/2): ")

        if choice == '1':
            data_type = DataType.INPUT
        elif choice == '2':
            data_type = DataType.TARGET
        else:
            print("Invalid choice, defaulting to Input.")
            data_type = DataType.INPUT
        config[field] = {'encoding': encoding, 'data_type': data_type}

    print("Python dictionary configuration:")
    print(config)
    return config

def create_categorical_dataset(data_folder_path, config=None, nrows=None, split=0.8):

    dataset_folder_path = os.path.join(data_folder_path, 'BC_DATA')
        
    if not os.path.exists(dataset_folder_path):
        raise FileNotFoundError(
            f"The file {dataset_folder_path} does not exist.")
    
    dataframe_path = os.path.join(dataset_folder_path, 'Data.xlsx')

    if not os.path.exists(dataframe_path):
        raise FileNotFoundError(
            f"The file {dataframe_path} does not exist.")
    
    dataframe = pd.read_excel(dataframe_path, nrows=nrows)  
    if config is None:
        config = create_breast_cancer_config(dataframe)

    processed_input_dataframe = pd.DataFrame()
    targets = None

    for field, settings in config.items():
        print(f"Processing field: {field} with settings: {settings}")

        encoding = settings['encoding']
        data_type = settings['data_type']
        processed_dataframe = pd.DataFrame()

        if data_type == DataType.TARGET:
            if targets is None:
                if encoding == EncodingType.RAW_INT:
                    targets = dataframe[field].astype('int').to_list()
                else:
                    raise ValueError("Target field must use RAW_INT encoding.")
            else:
                raise ValueError("Multiple target fields are not supported.")
        
        elif data_type == DataType.INPUT:
            if encoding == EncodingType.ONE_HOT:
                one_hot = pd.get_dummies(dataframe[field], prefix=field)
                processed_dataframe = pd.concat([processed_dataframe, one_hot], axis=1)

            elif encoding == EncodingType.INT_EMBEDDING:
                categories = dataframe[field].astype('category').cat.categories
                numbers = []
                for category in categories:
                    if type(category) == int:
                        numbers.append(category)
                    else:
                        match = re.search(r"\d+", str(category))
                        if match:
                            numbers.append(int(match.group()))
                        else:
                            raise ValueError(f"Cannot extract number from category: {category}")
                        
                sorted_categories = [x for _, x in sorted(zip(numbers, categories))]
                int_embedding = dataframe[field].astype(pd.CategoricalDtype(categories=sorted_categories, ordered=True))
                int_embedding = int_embedding.cat.codes + 1
                processed_dataframe = pd.concat([processed_dataframe, int_embedding.rename(field)], axis=1)

            elif encoding == EncodingType.RAW_INT:
                processed_dataframe = pd.concat([processed_dataframe, dataframe[field]], axis=1)

            else:
                raise ValueError(f"Unknown encoding type: {encoding}")
            
            processed_input_dataframe = pd.concat([processed_input_dataframe, processed_dataframe], axis=1)

        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
    if targets is None:
        raise ValueError("No target field specified in the configuration.")
    
    permutation = np.random.permutation(len(processed_input_dataframe))
    processed_input_dataframe = processed_input_dataframe.iloc[permutation].reset_index(drop=True)
    targets = [targets[i] for i in permutation]
    split_idx = int(len(processed_input_dataframe) * split)
    train_input_df = processed_input_dataframe.iloc[:split_idx].reset_index(drop=True)
    train_targets = targets[:split_idx]
    test_input_df = processed_input_dataframe.iloc[split_idx:].reset_index(drop=True)
    test_targets = targets[split_idx:]

    
    # save processed dataframes as numpy arrays to files
    info = {'input_columns': processed_input_dataframe.columns.tolist(),
            'classes': list(set(targets)),
            'config': config}
    
    complete_data = {'dataset': processed_input_dataframe.to_numpy(dtype=np.float16),
                     'targets': np.array(targets)}
    train_data = {'dataset': train_input_df.to_numpy(dtype=np.float16),
                  'targets': np.array(train_targets)}
    test_data = {'dataset': test_input_df.to_numpy(dtype=np.float16),
                 'targets': np.array(test_targets)}
    
    pickle.dump(info, open(os.path.join(dataset_folder_path, 'dataset_info.pkl'), 'wb'))
    np.savez_compressed(os.path.join(dataset_folder_path, 'complete_dataset.npz'), **complete_data)
    np.savez_compressed(os.path.join(dataset_folder_path, 'train_dataset.npz'), **train_data)
    np.savez_compressed(os.path.join(dataset_folder_path, 'test_dataset.npz'), **test_data)

    return config
    


class BreastCancerDataset(Dataset):
    def __init__(self, data_folder_path, split='complete'):
        dataset_folder_path = os.path.join(data_folder_path, 'BC_DATA')
        dataset_path = os.path.join(dataset_folder_path, f'{split}_dataset.npz')
        info_path = os.path.join(dataset_folder_path, 'dataset_info.pkl')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"The file {dataset_path} does not exist. Please create the dataset first.")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"The file {info_path} does not exist. Please create the dataset first.")
        
        data = np.load(dataset_path, allow_pickle=True)
        info = pickle.load(open(info_path, 'rb'))
        self.dataset = list(data['dataset'])
        self.targets = data['targets']
        self.input_columns = info['input_columns']
        self.classes = info['classes']
        self.config = info['config']

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float), int(self.targets[idx])

def split_by_age(dataset):
    age_idx = dataset.input_columns.index('Age recode with <1 year olds and 90+')
    upper_threshold = 4
    lower_threshold = 1

    subset_indices = []
    complement_indices = []
    for i in range(len(dataset)):
        age_value = dataset.dataset[i][age_idx]
        if lower_threshold <= age_value <= upper_threshold:
            subset_indices.append(i)
        else:
            complement_indices.append(i)
    return [torch.utils.data.Subset(dataset, subset_indices), torch.utils.data.Subset(dataset, complement_indices)]

if __name__ == "__main__":
    config = {'Age recode with <1 year olds and 90+': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Year of diagnosis': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Grade Recode (thru 2017)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Laterality': {'encoding': 'one_hot', 'data_type': 'input'}, 'Summary stage 2000 (1998-2017)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Marital status at diagnosis': {'encoding': 'one_hot', 'data_type': 'input'}, 'Rural-Urban Continuum Code': {'encoding': 'one_hot', 'data_type': 'input'}, 'ER Status Recode Breast Cancer (1990+)': {'encoding': 'one_hot', 'data_type': 'input'}, 'PR Status Recode Breast Cancer (1990+)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Chemotherapy recode (yes, no/unk)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Radiation recode': {'encoding': 'one_hot', 'data_type': 'input'}, 'Median household income inflation adj to 2022': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Sequence number': {'encoding': 'one_hot', 'data_type': 'input'}, 'Outcome': {'encoding': 'raw_int', 'data_type': 'target'}}

    # create_categorical_dataset("data", config=config, nrows=None)
    dataset = BreastCancerDataset("data", split='train')
    print(f"Dataset size: {len(dataset)}")
    print(f"Input size:{dataset.dataset[0].shape[0]}")

    print(dataset[0][0])  # Print first input sample
    print(dataset[0][1])  # Print first target sample
    print(f"Input columns: {dataset.input_columns}")
    print(f"Target classes: {dataset.classes}")

    subsets = split_by_age(dataset)
    print(f"Subset 1 size (age 1-4): {len(subsets[0])}")
    print(f"Subset 2 size (other ages): {len(subsets[1])}")