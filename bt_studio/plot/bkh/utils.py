#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import pandas as pd
import numpy as np

from bokeh.models import Range1d, ColumnDataSource


def on_resample(ns, df, origin, freq):
    # resmaple 生成连续日期
    # df = df.astype("float")
    df = df.replace('', np.nan).astype("float")

    if "datetime" in df.columns:
        origin.index = df.index = df.loc[:, "datetime"].apply(lambda x: pd.to_datetime(float(x), unit="s"))
    else:
        df.index = origin.index
        df.loc[:, "datetime"] = [x.timestamp() for x in df.index]

    if ns == "Feed":
        sample = df.resample(freq, closed="left", label="left").agg({
                'open': 'first',     
                'high': 'max',       
                'low': 'min',         
                'close': 'last',      
                'volume': 'mean',
                'datetime': 'max'
            })
    elif ns == "Strategy":
        sample = df.resample(freq, closed="left", label="left").max()
    else:
        sample = df.resample(freq, closed="left", label="left").last()

    sample.dropna(axis=0, how="all", inplace=True, subset=["datetime"])  

    if "datetime" in sample.columns: 
        sample.drop(columns=["datetime"], inplace=True)
    sample["datetime"] = sample.index
    return sample

def create_datasource(csv_path, freq):
    data = pd.read_csv(csv_path, header=1, sep=";")
    datasource = {}
    datasource["id"] = data.iloc[:,0].to_numpy()
    
    names_col = data.columns[1::2]
    datas_col = data.columns[2::2]
    for n_c, d_c in zip(names_col, datas_col):
        _src = {} 
        c_v = data.loc[:, d_c].astype(str)
        split_c_v = c_v.str.split(',', expand=True) 
        split_c_v.columns = d_c.split(',')
        resample_c_v = on_resample(n_c, split_c_v, data, freq)
        
        for key, v in resample_c_v.items():
            arr = v.to_numpy()
            # if key != "datetime":
            #     arr = np.where(np.isnan(arr), 0.0, arr)
            _src[key] = arr
        datasource[n_c] = ColumnDataSource(data=_src, name=n_c)
    return datasource, names_col

def merge_cds(*sources, on='datetime', dropna=True, exclude="datetime"):
    """
    使用 pandas 合并多个数据源
    
    参数:
    sources: 多个 ColumnDataSource 对象
    on: 合并的键列
    
    返回:
    合并后的 ColumnDataSource
    """
    tooltips = [("Date", "@datetime{%F}")]
    dfs = []
    for i, source in enumerate(sources):
        df = pd.DataFrame(source.data)

        subset = [col for col in df.columns if col != exclude]
        df.dropna(axis=0, how="all", inplace=True, subset=subset)

        chain_tooltip = [f" {key}: @{{{key}}}{{0.2f}}<br>" for key in source.column_names if  key != "datetime"]
        source_tooltip = ''.join(chain_tooltip) 
        tooltips.append((source.name, source_tooltip))
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        if on in merged_df.columns and on in df.columns:
            merged_df = pd.merge(merged_df, df, on=on, how='outer')
        else:
            merged_df = pd.concat([merged_df, df], axis=1)
    
    if on in merged_df.columns and pd.api.types.is_datetime64_any_dtype(merged_df[on]):
        merged_df = merged_df.sort_values(on)
    
    _source = ColumnDataSource(merged_df.reset_index(drop=True)) # remove index
    del _source.data["index"]
    return _source, tooltips


def on_shift(source, shift=1, drop=True):
    df = pd.DataFrame(source.data)
    if "datetime" in df.columns:
        df.set_index("datetime", drop=True, inplace=True)
    _df = df.shift(shift) # observer shift 1
    _df.dropna(axis=0, how="all", inplace=True) # drop last column / _df = _df.fillna(method="ffill")
    _df = _df.fillna(0.0) # keep sequence
    _df["datetime"] = _df.index
    _source = ColumnDataSource(_df.reset_index(drop=True))
    del _source.data["index"]
    return _source
