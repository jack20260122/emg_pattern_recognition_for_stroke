import os
import pandas as pd
import numpy as np
import glob
import re


def EMG_data_load():
    files1 = glob.glob('Stroke Patients Data/patient_1/*.xlsx')
    files2 = glob.glob('Stroke Patients Data/patient_2/*.xlsx')
    dirInf1 = files1 + files2

    feature_data = [None] * 400000
    label_data = [0] * 400000
    w = 30
    num = 0

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
                num += 1

                window_data = tmp_data[ii - w + 1:ii + 1]
                feature_data[num - 1] = window_data
                label_data[num - 1] = tmp_label[ii]

        print(file_path)

    feature_data = feature_data[:num]
    label_data = label_data[:num]

    EMGdata = {
        'feature_data': feature_data,
        'label_data': label_data
    }

    return EMGdata
