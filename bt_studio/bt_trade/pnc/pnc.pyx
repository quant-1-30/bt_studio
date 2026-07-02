# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdint cimport int32_t, int64_t
from libcpp.algorithm cimport lower_bound

from bt_core.sizers import _sizers
from bt_core.utils.dateintern import ts2intdt


cdef class TraderPlan:
    
    def __init__(self, bytes sid, double weight, bint isbuy, int32_t size=0, int32_t priority=1):
        self.core.sid = sid
        self.core.weight = weight
        self.core.isbuy = isbuy
        # small --> high
        self.core.size = size
        self.core.priority = priority 

    def __repr__(self):
        action = "BUY" if self.core.isbuy else "SELL"
        return f"[{action} (Pri:{self.core.priority})] {self.core.sid} -> {self.core.weight*100:.1f}%"

    def __lt__(self, object other): # .sort() 
        return self.core.priority < (<TraderPlan>other).core.priority


cdef class Pnc:
    """
        a. sell first 
        b. buy after 
    """

    def __init__(self, str sizer_name, **kwargs):
        self.interval = kwargs.pop("days_held", 5)
        self.stake = kwargs.pop("stake", 0.9)
        self.dd = kwargs.pop("dd", 0.25)
        self.trade_tick = kwargs.pop("pnc_time", 53700)
        self.sizer = _sizers[sizer_name](**kwargs)

        self._last_trade_day = 0

        self.pending_sells = {}
    
    # ==========================================================
    # TradingDays from benchmarkret
    # ==========================================================

    cpdef void set_trading_calendar(self, list days_list):
        self.v_trading_days.clear()
        cdef int32_t d
        for d in days_list:
            self.v_trading_days.push_back(d)

    cdef int32_t get_trading_days_held(self, int32_t created_day, int32_t current_day) noexcept nogil:
        if self.v_trading_days.empty():
            return 0
            
        cdef vector[int32_t].iterator it_created = lower_bound(self.v_trading_days.begin(), self.v_trading_days.end(), created_day)
        cdef vector[int32_t].iterator it_current = lower_bound(self.v_trading_days.begin(), self.v_trading_days.end(), current_day)
        
        return <int32_t>(it_current - it_created)

    # ==========================================================
    # risk control on tick
    # ==========================================================

    cpdef list check_risk(self, dict current_prices, object snapshot, dict stats):
        cdef double pnl
        cdef TraderPlan tmp
        cdef list positions = snapshot.positions, sells_by_risk=[]
        
        # ==========================================
        # 1. Macro Control --- Drawdown
        # ==========================================

        if stats["drawdown"].maxdd >= self.dd:
            print("reach maxdd and execute sell all")
            sells_by_risk = [TraderPlan(pos.sid, 1.0, False, pos.available, priority=1) for pos in positions if pos.size > 0]
            return sells_by_risk
        
        # ==========================================
        # 2. Asset Risk Control
        # ==========================================

        for pos in positions:
            sid = pos.sid
            
            if pos.size == 0 or sid in self.pending_sells:
                continue

            # PnL
            current_price = current_prices.get(sid, pos.cost_basis) 
            pnl = current_price / pos.cost_basis 

            if pnl <= self.stake: 
                tmp = TraderPlan(sid, 1.0, False, pos.available, priority=0) 
                sells_by_risk.append(tmp) 
                self.pending_sells[sid] = tmp
                
        return sells_by_risk
    
    # ==========================================================
    # generate execution plan
    # ==========================================================

    cpdef dict generate_plan(self, int64_t current_ts, dict topk_info, dict current_prices, object snapshot, dict stats): 
        # topk_info sort by score
        cdef bytes sid
        cdef int32_t days_held, current_day, slots, buy_count=0
        cdef double pnl, wgt_ratio
        cdef TraderPlan tmp
        
        cdef object pos
        cdef list positions = snapshot.positions , sells=[], buys=[]
        cdef object account = snapshot.account 
        cdef current_intday = ts2intdt(current_ts) 

        # ==========================================================
        # intraday seconds and days since 1970
        # ==========================================================
        # 14:55:00 ---> 53700
        cdef int64_t intraday_sec = (current_ts + 28800) % 86400
        
        # days ---> since 1970
        cdef int32_t current_day_id = (current_ts + 28800) // 86400 
        
        if intraday_sec < self.trade_tick:  
            return {"sell": [], "buy": []}

        if self._last_trade_day == current_day_id:
            return {"sell": [], "buy": []}
        
        s_wgt = self.sizer.getsizing(topk_info, snapshot, False)

        for pos in positions:
            sid = pos.sid

            if pos.size == 0 or sid in self.pending_sells:
                continue
            # ------------------------------------------
            # 1. HoldingDays and Conflict
            # ------------------------------------------
            days_held = self.get_trading_days_held(<int32_t>pos.created_dt, current_intday)
            
            if days_held < self.interval - 1:
                continue
                
            if sid in topk_info:
                continue
        
            wgt_ratio = s_wgt.get(sid, 1.0)
            tmp = TraderPlan(sid, wgt_ratio, False, pos.available, priority=1)
            sells.append(tmp)
            self.pending_sells[sid] = tmp

        sells.sort()

        # ==========================================
        # 2. Slot Control ---> Available Slots = TopK - (Positions - Sells)
        # ==========================================
        slots = len(topk_info) - (len(positions) - len(self.pending_sells))
        
        if slots <= 0:
            return {"sell": sells, "buy": []}

        # ==========================================
        # 3. Cash Control
        # ==========================================
         
        if account.cash <= 10000: # cash safety
            return {"sell": sells, "buy": []}
        
        # ==========================================
        # 4. Buy Control
        # ==========================================

        b_wgt = self.sizer.getsizing(topk_info, snapshot, True)

        for sid in topk_info: 
            if sid in self.pending_sells:
                continue

            wgt_ratio = b_wgt.get(sid, 0.0)
            if wgt_ratio > 0:
                tmp = TraderPlan(sid, wgt_ratio, True, 0, priority=1)
                buys.append(tmp)
            
            buy_count +=1
            if buy_count >= slots:
                break

        buys.sort()

        self._last_trade_day = current_day_id
        return {"sell": sells, "buy": buys}

    cpdef void on_updt(self, dict mtrades): # TradeBody
        cdef bytes sid
        cdef int32_t remain
        cdef TraderPlan tp
        cdef object trade

        for sid, trades in mtrades.items():
            executed_size = sum([trade.executed_size for trade in trades])

            if sid in self.pending_sells:
                tp = self.pending_sells[sid]
                remain = tp.core.size - executed_size
                if remain <= 0:
                    del self.pending_sells[sid]


# how to extend to multistrategy and MultiPnc


_pnc = {
 "default": Pnc
}
