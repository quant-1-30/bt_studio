    # # run live


class LiveMacroStateManager:
    def __init__(self, cache_file="/data/state/macro_history.json"):
        self.cache_file = cache_file
        # 里面只存了 252 个 float 数字
        self.history_returns = self._load_cache() 

    def get_today_macro_state(self, today_1455_close, yesterday_1455_close):
        # 1. 仅需计算今天的单日收益率
        today_ret = (today_1455_close / yesterday_1455_close) - 1.0
        
        # 2. 从缓存中获取历史 252 天数据，瞬间算出 p20, p80
        p20 = np.percentile(self.history_returns, 20)
        p80 = np.percentile(self.history_returns, 80)
        
        # 3. 判定今天状态
        if today_ret < p20: macro_state = 0
        elif today_ret > p80: macro_state = 2
        else: macro_state = 1
        
        # 4. 增量更新：把今天的加进去，把最老的一天踢掉，保存回磁盘/Redis
        self.history_returns.append(today_ret)
        if len(self.history_returns) > 252:
            self.history_returns.pop(0)
        self._save_cache()
        
        return macro_state


class LiveGPDManager:
    def __init__(self, cache_file="/data/state/gpd_state.json"):
        # 缓存里存着：上一次更新的月份、当前的 edges、当前的 centers、历史收益池
        self.state = self._load_cache() 

    def get_today_gpd(self, today_date_int, today_all_market_rets):
        year = today_date_int // 10000
        month = (today_date_int % 10000) // 100
        current_month_idx = year * 12 + month
        
        # 把今天的全市场真实收益率追加进池子 (每天只追加几百个 float)
        self.state["history_pool"].extend(today_all_market_rets)
        # 维持窗口大小，踢掉老数据
        # ...
        
        # 🌟 核心：判定今天是否该更新了？
        # 如果还没到半年，直接 O(1) 返回上一次算好的边界！
        if current_month_idx - self.state["last_update_month"] < 6:
            return self.state["edges"], self.state["centers"]
            
        # 如果刚好跨越了半年，今天触发一次重算 (耗时不到 1 秒)
        edges, centers = calculate_gpd(self.state["history_pool"], quantiles=[0.1, 0.3, 0.7, 0.9])
        self.state["edges"] = edges
        self.state["centers"] = centers
        self.state["last_update_month"] = current_month_idx
        self._save_cache()
        
        return edges, centers


def live_trading_1455_task(today_date_int):
    """ 实盘每天 14:55 准时唤醒的定时任务 """
    
    # 1. 极速获取今天的宏观状态 (耗时 1ms)
    macro_state = live_macro_manager.get_today_macro_state(...)
    
    # 2. 极速获取今天的 GPD 参数 (99%的时间直接读缓存，耗时 0.1ms)
    edges, centers = live_gpd_manager.get_today_gpd(...)
    
    # 3. 提取全市场今天截至 14:55 的特征
    # 注意：只请求今天的 Tick 数据！最多拉取今天一天的量，数据量极小！
    today_tick_data = get_today_ticks() 
    today_features = extract_today_asset_features(today_tick_data)
    
    # 4. 横截面扫描与打分 (耗时 2秒)
    for asset in today_features:
        # 用 DTW 算距离，用 global_fsm 算分数
        pass 
        
    # 5. Top-K 截面排序下单
    place_orders(top_10_stocks)
    
    # 6. 【在线学习】：核算昨天触发信号的真实收益，更新 FSM 大脑
    # (昨天买了啥，今天收盘收益是多少，直接增量更新 FSM 矩阵，然后序列化存入 Redis)
    update_fsm_with_yesterday_trades()

    # def daily_live_job(today_date):
    #     # 1. 唤醒记忆
    #     state = load_pickle("/models/last_valid_state.pkl")
    #     live_model = MotifFSMModel(state["config"], macro_dict, gpd_dict)
    #     live_model.learned_motif = state["motif"]
    #     live_model.fsm_prior_matrix = state["fsm_prior"]
        
    #     # 2. 只需要提昨天和今天两天的特征！
    #     recent_panel = fetch_recent_two_days_features(today_date)
        
    #     # 3. 注入昨天的实盘记忆 (从 Redis 或 本地 JSON 读)
    #     live_model.pending_yesterday_top_k = load_json("/models/yesterday_signals.json")
        
    #     # 4. 单步执行 (核心代码完全复用！)
    #     today_signals, new_fsm_matrix = live_model.predict_oos_chronological(recent_panel, top_k=10)
        
    #     # 5. 推送交易信号
    #     send_to_trading_engine(today_signals)
        
    #     # 6. 每日休眠，保存进度
    #     state["fsm_prior"] = new_fsm_matrix
    #     save_pickle(state, "/models/last_valid_state.pkl")
    #     save_json(today_signals, "/models/yesterday_signals.json")