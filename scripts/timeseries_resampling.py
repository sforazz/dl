import pandas as pd
import argparse
import numpy as np


def ts_resampling(f, sheet, os, ns):

    outname = f.split('.xlsx')[0]+'_sheet_{}_resampled_{}minutes'.format(sheet, ns)
    xls = pd.ExcelFile(f)
    try:
        dataframe = pd.read_excel(xls, sheet-1)
    except IndexError:
        raise Exception('The specified sheet number, {0}, is out of range. The number of sheets in this excel file is: {1}'
                        .format(sheet, len(xls.sheet_names)))
    col_headers = list(dataframe.columns.values)
    data = [dataframe[x].tolist() for x in col_headers if type(dataframe[x].tolist()[0])==np.float and not np.isnan(dataframe[x].tolist()[0])]
    for i, d in enumerate(data):
        data_corrected = [x for x in d if type(x)!=str]
        index = pd.date_range('1/1/2000', periods=len(data_corrected), freq='{}T'.format(os))
        series = pd.Series(data_corrected, index=index)
        new_series = series.resample('{}T'.format(ns)).mean()
        new_series.to_excel(outname+'_col{}.xlsx'.format(i+1))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, help='Input excel file.')
    parser.add_argument('--sheet', '-s', type=int, default=1, help='Number of the sheet containing the data to be resampled.'
                        ' Default is 1 (the first one).')
    parser.add_argument('--original_sampling', '-os', type=int, default=2,
                        help='Sampling rate (in minutes) of the original time series. Default is 2 (minutes).')
    parser.add_argument('--new_sampling', '-ns', type=int, default=5,
                        help='Sampling rate (in minutes) of the resampled time series. Default is 5 (minutes).')

    cliargs, unknown = parser.parse_known_args()

    artefacts = ts_resampling(cliargs.file, cliargs.sheet, cliargs.original_sampling, cliargs.new_sampling)

    print('Done!')