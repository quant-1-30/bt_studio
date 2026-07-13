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
import bt_core as bt

from typing import List, Dict, Any

from bt_protocol._protocol import SnapshotBody


class FixedSize(bt.Sizer):
    '''
    This sizer simply returns a fixed size for any operation.
    Size can be controlled by specifying the ``stake`` parameter.
    '''
    def __init__(self, stake: float = 0.9, **kwargs):
        # super().__init__()
        self.stake = stake

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if isbuy:
            ratio = self.stake / len(topk_info) if len(topk_info) > 1 else self.stake
            _sizer = {sid: ratio for sid in topk_info.keys()}
        else:
            _sizer = {p.sid: 1.0 for p in snapshot.positions if p.size > 0}
        return _sizer


class WeightedSizer(bt.Sizer):
    '''
    This sizer will return a size based on the relative scores of the
    candidates. The scores should be provided by the strategy in the
    ``sizing_context`` attribute as a dict of {sid: score}. The sizer will
    then calculate the size for each candidate as:
    
    size(sid) = max(score(sid), 0) / sum(max(score(sid_i), 0) for sid_i in candidates)
    
    This means that only candidates with positive scores will receive allocation,
    and the allocation will be proportional to their score relative to the total
    positive score of all candidates.
    '''
    def __init__(self, stake: float = 0.9, **kwargs):
        self.stake = stake

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if isbuy:
            if not topk_info:
                return {}
            
            total_score = sum(max(info.get("score", 0.0), 0.0) for info in topk_info.values())
            
            if total_score <= 0:
                return {sid: 0.0 for sid in topk_info.keys()}
                
            _sizer = {
                sid: self.stake * max(info.get("score", 0.0), 0.0) / total_score 
                for sid, info in topk_info.items()
            }
        else:
            _sizer = {p.sid: 1.0 for p in snapshot.positions if p.size > 0}
        return _sizer


class TurtleSizer(bt.Sizer):
    '''This sizer implements the position sizing method used in the Turtle Trading system, 
       which is based on the Average True Range (ATR) of the asset. The size is calculated as:  
         size_ratio = risk_ratio / atr_ratio
         
         where:
            - cash is the current available cash in the broker
            - risk_ratio is the percentage of cash to risk on each trade (e.g., 0.01 for 1%)
            - atr is the Average True Range of the asset, which measures its volatility
            - close is the current closing price of the asset
    '''
    def __init__(self, stake: float = 0.9, risk_ratio: float = 0.01, **kwargs):
        self.stake = stake
        self.risk_ratio = risk_ratio

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if isbuy:
            _sizer = {
                sid: self.stake * self.risk_ratio * (info.get("close", 0.0) / info.get("atr", 1.0)) 
                for sid, info in topk_info.items()
            }
        else:
            _sizer = {p.sid: 1.0 for p in snapshot.positions if p.size > 0}
        return _sizer


class KellySizer(bt.Sizer):
    '''This sizer will return a size based on the Kelly Criterion
 
     The Kelly Criterion is a formula used to determine the optimal size of a
     series of bets in order to maximize the logarithm of wealth. It is often
     used in gambling and investing to help manage risk and maximize returns.
 
     The formula for the Kelly Criterion is:
 
     f* = (bp - q) / b
 
     where:
 
       - f* is the fraction of the current bankroll to wager
 
       - b is the net odds received on the wager (i.e., "b to 1") - this is
         calculated as (1 / (price - 1)) for buy operations and (1 / price) for
         sell operations
 
       - p is the probability of winning (i.e., the probability that the bet
         will pay off)
 
       - q is the probability of losing, which is equal to 1 - p
 
     The Kelly Criterion suggests that you should bet a fraction of your
     bankroll equal to f* in order to maximize your long-term growth rate. If
     f* is negative, it means that you should not place the bet at all.
 
     Note that the Kelly Criterion assumes that you have an edge over the house
     or market, meaning that your probability of winning (p) is greater than
     your probability of losing (q). If you do not have an edge, then betting
     according to the Kelly Criterion may lead to losses over time.
 
     This sizer requires that ``self.strategy`` has two methods implemented:
 
       - ``kelly_p(self, data)``: returns the probability of winning for the
         given data
 
       - ``kelly_q(self, data)``: returns the probability of losing for the
         given data
    '''
    def __init__(self, stake: float = 0.9, strategy_name: str = "", **kwargs):
        super().__init__()
        self.stake = stake
        self.strategy_name = strategy_name

    def _getsizing(self, topk_info: Dict[bytes, Any], snapshot: SnapshotBody, isbuy: bool):
        if isbuy:
            _sizer = {}
        else:
            _sizer = {p.sid: 1.0 for p in snapshot.positions if p.size > 0}
        return _sizer
