from libc.stdint cimport int32_t, int64_t
from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string


cdef struct CoreData:
    cpp_string sid
    double weight
    bint isbuy
    int32_t size 
    int32_t priority


cdef class TraderPlan:
    
    cdef readonly CoreData core


cdef class Pnc:
    cdef vector[int32_t] v_trading_days

    cdef int32_t interval
    cdef int32_t _last_trade_day
    cdef int32_t trade_tick
    cdef double stake
    cdef double dd
    cdef object sizer

    cdef list sells
    cdef list buys
    cdef dict pending_sells
    
    cpdef void set_trading_calendar(self, list days_list)

    cdef int32_t get_trading_days_held(self, int32_t created_day, int32_t current_day) noexcept nogil
    
    cpdef list check_risk(self, dict current_prices, object snapshot, dict stats)
    
    cpdef dict generate_plan(self, int64_t current_ts, dict topk_info, dict current_prices, object snapshot, dict stats) 

    cpdef void on_updt(self, dict sell_trades)

