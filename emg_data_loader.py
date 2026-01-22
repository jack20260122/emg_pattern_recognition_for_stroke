import os
import pandas as pd
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader


class EMGDataset(Dataset):

    def __init__(self, data_dir='Stroke Patients Data'):
        self.feature_data, self.label_data = self._load_data(data_dir)

    def _load_data(self, data_dir):

        files1 = glob.glob(os.path.join(data_dir, 'patient_1/*.xlsx'))
        files2 = glob.glob(os.path.join(data_dir, 'patient_2/*.xlsx'))
        dirInf1 = files1 + files2

        feature_data = []
        label_data = []
        w = 30

        for file_path in dirInf1:

            df = pd.read_excel(file_path, header=None)
            Z1 = df.values.tolist()

            I1 = []
            for j in range(len(Z1[0])):
                cell_value = Z1[0][j] if j < len(Z1[0]) else None
                if (cell_value is not None and
                        not (isinstance(cell_value, float) and np.isnan(cell_value)) and
                        re.search(r'^day', str(cell_value), re.IGNORECASE)):
                    I1.append(j)

            for j in I1:

                tmp_data = []
                tmp_label = []

                for row_idx in range(2, len(Z1)):
                    if row_idx < len(Z1):
                        row_data = Z1[row_idx]

                        feature_row = []
                        for col_idx in range(j, min(j + 8, len(row_data))):
                            feature_row.append(row_data[col_idx])
                        tmp_data.append(feature_row)

                        if j + 8 < len(row_data):
                            tmp_label.append(row_data[j + 8])
                        else:
                            tmp_label.append(None)

                ii = len(tmp_data)
                for k in range(len(tmp_data)):
                    if len(tmp_data[k]) > 0 and (tmp_data[k][0] is None or
                                                 (isinstance(tmp_data[k][0], float) and np.isnan(tmp_data[k][0]))):
                        ii = k
                        break

                tmp_data = tmp_data[:ii]
                tmp_label = tmp_label[:ii]

                for ii in range(w - 1, len(tmp_data)):
                    window_data = tmp_data[ii - w + 1:ii + 1]
                    feature_data.append(window_data)
                    label_data.append(tmp_label[ii])

            print(f"Processed: {file_path}")

        return feature_data, label_data

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset with length {len(self)}")

        features = torch.tensor(self.feature_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.float32) if self.label_data[
                                                                               idx] is not None else torch.tensor(0.0,
                                                                                                                  dtype=torch.float32)

        return features, label


def get_emg_dataloader(data_dir='Stroke Patients Data', batch_size=32, shuffle=True, num_workers=48):
    dataset = EMGDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":

    emg_dataloader = get_emg_dataloader(batch_size=16, shuffle=True)

    for batch_idx, (features, labels) in enumerate(emg_dataloader):
        print(f"Batch {batch_idx}: features shape = {features.shape}, labels shape = {labels.shape}")

        if batch_idx >= 2:
            break

    print(f"Total samples: {len(emg_dataloader.dataset)}")
