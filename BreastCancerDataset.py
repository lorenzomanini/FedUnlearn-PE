import pandas as pd
from tqdm import tqdm
import os
import re

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


class BreastCancerDataset(Dataset):
    def __init__(self, file_path, config=None, nrows=None):
        self.file_path = file_path
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The file {self.file_path} does not exist.")

        self.dataframe = pd.read_excel(self.file_path, nrows=nrows)

        if config is None:
            config = create_breast_cancer_config(self.dataframe)
        self.config = config

        self.dataset, self.targets, self.targets_classes = self.create_dataset()

    def create_dataset(self):
        processed_input_dataframe = pd.DataFrame()
        processed_target_dataframe = pd.DataFrame()

        for field, settings in self.config.items():
            print(f"Processing field: {field} with settings: {settings}")

            encoding = settings['encoding']
            data_type = settings['data_type']
            processed_dataframe = pd.DataFrame()

            if encoding == EncodingType.ONE_HOT:
                one_hot = pd.get_dummies(self.dataframe[field], prefix=field)

                processed_dataframe = pd.concat(
                    [processed_dataframe, one_hot], axis=1)
            elif encoding == EncodingType.INT_EMBEDDING:
                categories = self.dataframe[field].astype(
                    'category').cat.categories
                
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
                        
                sorted_categories = [
                    x for _, x in sorted(zip(numbers, categories))]
                int_embedding = self.dataframe[field].astype(
                    pd.CategoricalDtype(categories=sorted_categories, ordered=True))
                int_embedding = int_embedding.cat.codes + 1

                processed_dataframe = pd.concat(
                    [processed_dataframe, int_embedding.rename(field)], axis=1)
            elif encoding == EncodingType.RAW_INT:
                processed_dataframe = pd.concat(
                    [processed_dataframe, self.dataframe[field]], axis=1)
            else:
                raise ValueError(f"Unknown encoding type: {encoding}")
            
            if data_type == DataType.INPUT:
                processed_input_dataframe = pd.concat(
                    [processed_input_dataframe, processed_dataframe], axis=1)
            elif data_type == DataType.TARGET:
                processed_target_dataframe = pd.concat(
                    [processed_target_dataframe, processed_dataframe], axis=1)
        
        classes = processed_target_dataframe.nunique().tolist()

        return processed_input_dataframe.to_numpy(dtype=float), processed_target_dataframe.to_numpy(dtype=float), classes
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float), torch.tensor(self.targets[idx], dtype=torch.float)


if __name__ == "__main__":
    file_path = r"data\Data4.xlsx"
    config = {'Age recode with <1 year olds and 90+': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Year of diagnosis': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Grade Recode (thru 2017)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Laterality': {'encoding': 'one_hot', 'data_type': 'input'}, 'Summary stage 2000 (1998-2017)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Marital status at diagnosis': {'encoding': 'one_hot', 'data_type': 'input'}, 'Rural-Urban Continuum Code': {
        'encoding': 'one_hot', 'data_type': 'input'}, 'ER Status Recode Breast Cancer (1990+)': {'encoding': 'one_hot', 'data_type': 'input'}, 'PR Status Recode Breast Cancer (1990+)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Chemotherapy recode (yes, no/unk)': {'encoding': 'one_hot', 'data_type': 'input'}, 'Radiation recode': {'encoding': 'one_hot', 'data_type': 'input'}, 'Median household income inflation adj to 2022': {'encoding': 'int_embedding', 'data_type': 'input'}, 'Outcome': {'encoding': 'one_hot', 'data_type': 'target'}}

    dataset = BreastCancerDataset(file_path, config=config, nrows=10000)
    print(f"Dataset size: {len(dataset)}")
    print(f"Input size:{dataset.dataset.shape[1]}, Target size:{dataset.targets.shape[1]}")
