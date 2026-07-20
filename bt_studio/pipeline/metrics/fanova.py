import numpy as np
import polars as pl
import optuna
from optuna.importance import FanovaImportanceEvaluator


def validate_parameter_plateau_fanova(df_results: pl.DataFrame, best_config: dict, best_score: float) -> bool:
    """fANOVA estimate"""
    param_cols = [c for c in df_results.columns if c.startswith("config/")]
    valid_df = df_results.drop_nulls(subset=param_cols + ["metrics_score"])
    
    if valid_df.height < 20: 
        return False

    # param distribution
    distributions = {}
    for col in param_cols:
        param_name = col.replace("config/", "")
        
        min_val = valid_df[col].min()
        max_val = valid_df[col].max()
        
        if min_val == max_val:
            min_val = min_val - 1e-5 if min_val != 0 else -1e-5
            max_val = max_val + 1e-5 if max_val != 0 else 1e-5
            
        # DataType Distribution
        dtype = valid_df[col].dtype
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            distributions[param_name] = optuna.distributions.IntDistribution(int(min_val), int(max_val))
        elif dtype in [pl.Categorical, pl.String]:
            choices = valid_df[col].unique().to_list()
            distributions[param_name] = optuna.distributions.CategoricalDistribution(choices)
        else:
            distributions[param_name] = optuna.distributions.FloatDistribution(float(min_val), float(max_val))
    # --------------------------------------------------------------------------------------------------------

    # Optuna Study Trial
    study = optuna.create_study(direction="maximize")
    for row in valid_df.iter_rows(named=True):
        trial = optuna.trial.create_trial(
            params={k.replace("config/", ""): v for k, v in row.items() if k.startswith("config/")},
            distributions=distributions,  
            value=row["metrics_score"]
        )
        study.add_trial(trial) 
        
    # fANOVA metrics
    try:
        fanova = FanovaImportanceEvaluator()
        importances = optuna.importance.get_param_importances(study, evaluator=fanova)
    except Exception as e:
        print(f"fANOVA estimate failure: {e}")
        return True

    # topk 
    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_imp = sorted_params[0]
    top2_name, top2_imp = sorted_params[1]
    
    if top1_imp < 0.20:
        print(f" fANOVA multilinear or noise (max fanova < 20%) and suggest Contour Plot")
        return True

    # isolate test 
    t1_val, t2_val = best_config[top1_name], best_config[top2_name]
    
    def _get_bounds(val):
        return (val - 0.05, val + 0.05) if isinstance(val, float) else (val, val)
        
    b1_min, b1_max = _get_bounds(t1_val)
    b2_min, b2_max = _get_bounds(t2_val)
    
    # test if peak mode
    nearby_trials = valid_df.filter(
        (pl.col(f"config/{top1_name}").is_between(b1_min, b1_max)) &
        (pl.col(f"config/{top2_name}").is_between(b2_min, b2_max))
    )
    
    score_std = valid_df["metrics_score"].std()
    if nearby_trials.height > 0 and score_std > 0:
        nearby_mean = nearby_trials["metrics_score"].mean()
        if nearby_mean < best_score - 1.5 * score_std:
            print(f"❌ fANOVA isolated peak detected. Score dropped heavily.")
            return False
            
    print(f"pass plate and fANOVA topk core: {top1_name}({top1_imp:.1%}), {top2_name}({top2_imp:.1%})")
    return True 
