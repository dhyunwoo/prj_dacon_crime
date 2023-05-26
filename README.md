# prj_dacon_crime
---
## TEST 1 2023-05-22
model: LGBMClassifier
train_test_split(X, y, test_size=0.2)
Best Score: 0.9591628511572694
Best trial: {'reg_alpha': 7.526669770455994e-06, 'reg_lambda': 0.007315347513554999, 'max_depth': 20, 'num_leaves': 158, 'colsample_bytree': 0.49294962342814835, 'subsample': 0.940374807411319, 'subsample_freq': 2, 'min_child_samples': 26, 'max_bin': 419}
Final Result: 0.3365037834

def objective(trial: Trial) -> float:
    params_lgb = {
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }
        

    model = LGBMClassifier(**params_lgb)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        early_stopping_rounds=5,
        verbose=False,
    )

    lgb_pred = model.predict_proba(X_valid)
    log_score = log_loss(y_valid, lgb_pred)
    
    return log_score
    
   --- 
## TEST 2 2023-05-23
model: XGBClassifier
train_test_split(X, y, test_size=0.2)
Accuracy: 0.526120884675999
 Best hyperparameters: 
    max_depth: 8
    learning_rate: 0.08219516566928238
    n_estimators: 345
    min_child_weight: 6
    gamma: 0.08670609931022182
    subsample: 0.6043181515340048
    colsample_bytree: 0.6125394488914258
    reg_alpha: 0.0008601779195278824
    reg_lambda: 0.0746891279586882
Final Result: 0.3300927952

def RF_objective(trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }

    # Fit the model
    model = xgb.XGBClassifier(**params)
    
    BOOST = model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose = False)
    
    # Make predictions
    y_pred = BOOST.predict(X_valid)
    
    # Evaluate predictions
    F1_SCORE = f1_score(y_valid, y_pred, average="macro")

    return F1_SCORE
    
---   
## TEST 3 2023-05-26
model: LGBMClassifier
train_test_split(X, y, test_size=0.2)
Final Result: 0.4158992664
