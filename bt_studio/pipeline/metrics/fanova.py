import numpy as np
import polars as pl
import optuna
from optuna.importance import FanovaImportanceEvaluator


def validate_parameter_plateau_fanova(df_results: pl.DataFrame, best_config: dict, best_score: float) -> bool:
    """fANOVA"""
    param_cols = [c for c in df_results.columns if c.startswith("config/")]
    
    valid_df = df_results.filter(pl.col("metrics_score") > -500.0).drop_nulls(subset=param_cols + ["metrics_score"])
    
    if valid_df.height < 20: 
        print("Effective Trial less than 20  and Reject")
        return False

    # Optuna ...
    distributions = {}
    for col in param_cols:
        param_name = col.replace("config/", "")
        min_val, max_val = valid_df[col].min(), valid_df[col].max()
        if min_val == max_val:
            min_val = min_val - 1e-5 if min_val != 0 else -1e-5
            max_val = max_val + 1e-5 if max_val != 0 else 1e-5
            
        dtype = valid_df[col].dtype
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            distributions[param_name] = optuna.distributions.IntDistribution(int(min_val), int(max_val))
        elif dtype in [pl.Categorical, pl.String]:
            choices = valid_df[col].unique().to_list()
            distributions[param_name] = optuna.distributions.CategoricalDistribution(choices)
        else:
            distributions[param_name] = optuna.distributions.FloatDistribution(float(min_val), float(max_val))

    study = optuna.create_study(direction="maximize")
    for row in valid_df.iter_rows(named=True):
        trial = optuna.trial.create_trial(
            params={k.replace("config/", ""): v for k, v in row.items() if k.startswith("config/")},
            distributions=distributions,  
            value=row["metrics_score"]
        )
        study.add_trial(trial) 
        
    try:
        fanova = FanovaImportanceEvaluator()
        importances = optuna.importance.get_param_importances(study, evaluator=fanova)
    except Exception as e:
        print(f"❌ fANOVA Calculate Failure: {e}")
        return False # 

    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_imp = sorted_params[0]
    top2_name, top2_imp = sorted_params[1]
    
    # 15% 
    def _get_relative_bounds(pname, val):
        col_vals = valid_df[f"config/{pname}"].to_numpy()
        p_range = col_vals.max() - col_vals.min()
        delta = p_range * 0.15 if p_range > 0 else 0.5
        return (val - delta, val + delta)
        
    b1_min, b1_max = _get_relative_bounds(top1_name, best_config[top1_name])
    b2_min, b2_max = _get_relative_bounds(top2_name, best_config[top2_name])
    
    nearby_trials = valid_df.filter(
        (pl.col(f"config/{top1_name}").is_between(b1_min, b1_max)) &
        (pl.col(f"config/{top2_name}").is_between(b2_min, b2_max)) &
        (pl.col("metrics_score") < best_score - 1e-5)
    )
    
    # IQR avoid std 
    scores = valid_df["metrics_score"].to_numpy()
    q75, q25 = np.percentile(scores, [75, 25])
    iqr_scale = max(q75 - q25, 1e-4)
    
    if nearby_trials.height > 0:
        worst_neighbor = nearby_trials["metrics_score"].min()
        if (best_score - worst_neighbor) > 1.5 * iqr_scale:
            print(f"❌ fANOVA Peak Score: {best_score:.2f}, adjecent Low Score: {worst_neighbor:.2f} > 1.5xIQR)")
            return False
            
    print(f"✅ fANOVA Passed: {top1_name}({top1_imp:.1%}), {top2_name}({top2_imp:.1%})")
    return True 
