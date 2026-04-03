# ---- Minimal logging + GPBO with per-trial save of the best-NLL run ----
import numpy as np, pandas as pd, random, math, os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.stats import norm

FEAT_COL, DELTA_COL = 'Changed feature index', 'Changed amount'
xai_idx = {v:k for k,v in XAI_types.items()}

def _get_p(fd): return fd.get('selected_p', fd.get('p_selected', 0.0))
def _nll_for_choice(out_probs, chosen_feat, lapse, num_features=6):
    sm = smooth_probs_with_lapse(out_probs, lapse=lapse, num_features=num_features)
    p = max(min(float(_get_p(sm.get(chosen_feat, {}))), 1-1e-9), 1e-9)
    return -math.log(p)

def _ei(mu, sigma, best, xi=0.01):
    imp = best - mu - xi
    Z = np.divide(imp, sigma, out=np.zeros_like(mu), where=(sigma>0))
    return np.where(sigma>0, imp*norm.cdf(Z) + sigma*norm.pdf(Z), 0.0)

# Pass the EXACT training_cog_params you used to train PPO
def make_obs_builder(training_cog_params, strategies, XAI_types):
    # order of varied params in env: all (lo,hi) except 'chi', in dict order
    varied_param_names = []
    for k, v in training_cog_params.items():
        if k == "chi": 
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            varied_param_names.append(k)

    # map strings -> keys used by env (0:'DT', 1:'LR', 2:'DT+LR')
    xai_key_from_name = {v:k for k,v in XAI_types.items()}
    xai_key_from_name_shown = xai_key_from_name  # same map

    def build_obs(curr_chi, step_idx, with_xai, condition_name, shown_name,
                  counts, success_rates, mean_times, current_cog_params):
        # condition and shown are KEYS in env obs
        cond_key  = float(xai_key_from_name[condition_name])
        shown_key = float(xai_key_from_name_shown[shown_name])

        obs = [float(curr_chi), float(step_idx), float(with_xai), cond_key, shown_key]

        # triplets per strategy id order
        for sid in strategies.keys():  # assumes stable int keys 0..K-1
            obs += [
                float(counts.get(sid, 0)),
                float(success_rates.get(sid, 0.0)),
                float(mean_times.get(sid, 0.0)),
            ]

        # tail: varied params in EXACT training order
        for name in varied_param_names:
            obs.append(float(current_cog_params.get(name, 0.0)))

        return np.array(obs, dtype=np.float32), varied_param_names

    return build_obs, varied_param_names


def score_participant_with_theta(model, user_loader, participant_id, ai_dataset_loader,
                                 lr_df, dt_df, metadata_df, theta, lapse=0.1):
    rt, over_margin, chi = theta['retrieval_threshold'], theta['over_margin'], theta['chi']
    info = user_loader.get_participant_info(participant_id)
    app_id, model_name = info['app_id'], info['model']
    condition, complexity = info['condition'], info['complexity']
    ai_loader = filter_by_app_and_model(ai_dataset_loader, app_id, model_name)
    bounds = ai_loader.get_bounds_for_app(app_id)
    transform, ai = load_transform_and_ai(app_id)

    lr_exp = LogisticRegressionInterpreter(lr_df, metadata_df, app_id, "mlp",
                                           variant=("sparse" if complexity=="low" else "dense"))
    dt_exp = DecisionTreeInterpreter(dt_df, metadata_df, app_id, "mlp",
                                     depth=(3 if complexity=='high' else 2))

    mem_lr = _make_memory(retrieval_threshold=rt, latency_factor=0.2)
    mem_dt = _make_memory(retrieval_threshold=rt, latency_factor=0.2)

    fwd = user_loader.get_forward_trials(participant_id)
    cf  = user_loader.get_counterfactual_trials(participant_id)

    add_lr_heuristic_to_memory(lr_exp, mem_lr)
    add_dt_to_memory(mem_dt, dt_exp)

    # prime memory from forward (LR heuristic refresh if with-XAI)
    for _, tr in fwd.iterrows():
        insts, _ = ai_loader.load_instances([tr['Instance Id']], normalize=True)
        x = insts[0]
        _, t, inf = lr_heuristic(x, mem_lr, lr_exp, T_enc=2, ddm_a=1.0, ddm_s=0.8)
        if str(tr['Tested w/ XAI']).lower().startswith('w'):
            refresh_lr_heuristic_in_memory(mem_lr, lr_exp, inf, actual=tr['AI prediction'])
        mem_lr.tick(t)

    for _, tr in fwd.iterrows():
        insts, _ = ai_loader.load_instances([tr['Instance Id']], normalize=False)
        x = insts[0]
        mode = 'read' if tr['Tested w/ XAI']=='w/ XAI' else 'retrieve'
        dt_traverse(x, mem_dt, dt_exp, mode=mode)
        if mode=='read':
            refresh_dt_path_in_memory(mem_dt, dt_exp, x)
        mem_dt.tick(t)

    counts = {k:0 for k in strategies.keys()}
    succ   = {k:0.0 for k in strategies.keys()}
    mtime  = {k:0.0 for k in strategies.keys()}
    step   = 0
    cond_idx = xai_idx[condition]

    trials_log = []
    NLLs, MAEs, TIMES = [], [], []

    obs_builder, varied_param_names = make_obs_builder(training_cog_params, strategies, XAI_types)

    current_cog_params = training_cog_params.copy()
    current_cog_params = {
        'retrieval_threshold': theta['retrieval_threshold'],
        'lapse': lapse,
        'over_margin': theta['over_margin'],
    }


    for ti, tr in cf.iterrows():
        iid, ai_pred = tr['Instance Id'], tr['AI prediction']
        feat_chosen  = f'a{tr[FEAT_COL]}'; delta_chosen = float(tr[DELTA_COL])
        (x_raw,_), (x_norm,_) = ai_loader.load_instances([iid], normalize=False), ai_loader.load_instances([iid], normalize=True)
        x_raw, x_norm = x_raw[0], x_norm[0]

        with_xai = 1 if tr['Tested w/ XAI'] in (1,'w/ XAI','with XAI','With XAI') else 0
        shown = tr.get('XAIType', None) if condition=='DT+LR' else condition
        if shown is None: shown = random.choice(['DT','LR']) if condition=='DT+LR' else condition
        shown_idx = xai_idx['DT'] if shown=='DT' else xai_idx['LR']

        obs, _ = obs_builder(
            curr_chi=theta['chi'],
            step_idx=step,
            with_xai=with_xai,
            condition_name=condition,
            shown_name=shown,
            counts=counts, success_rates=succ, mean_times=mtime,
            current_cog_params=current_cog_params
        )
        action, _ = model.predict(obs, deterministic=True)
        # print(action)
        strat_id, depth = int(action[0]), int(action[1]); strat = strategies[strat_id]
        mode = ('read' if with_xai else 'retrieve')

        
        if condition=='DT':
            if strat in ('zero_out_lr_displayed', 'zero_out_lr_heuristic', 'recall_change_lr'):
                strat = 'change_path_dt'  # fallback
        elif condition=='LR':
            if strat in ('change_path_dt', 'recall_change_dt'):
                strat = 'zero_out_lr_heuristic'  # fallback
        elif shown=='DT':
            if strat in ('zero_out_lr_displayed'):
                strat = 'change_path_dt'  # fallback


        # get feature probs + mean_delta (+time)
        if strat == "change_path_dt":
            out = cf_change_path_dt(x_raw, dt_exp, bounds, memory=mem_dt, chosen_depth=depth, mode=mode, tau=1.0)
            t = out.pop('expected_time', 0.0); mem_dt.tick(t)
            if int(dt_exp.apply_to_instance(x_raw)['class_index']) != ai_pred:
                for k in out: out[k]['mean_delta'] *= -1
        elif strat == "zero_out_lr_heuristic":
            out = cf_lr_heuristic(x_norm, mem_lr, lr_exp, bounds, K_top=6, y_actual=ai_pred)
            t = out.pop('expected_time', 0.0); mem_lr.tick(t)
        elif strat == "zero_out_lr_displayed":
            out = cf_lr_calculation(x_raw, lr_exp, bounds=bounds, memory=mem_lr)
            t = out.pop('expected_time', 0.0); mem_lr.tick(t)
            if int(lr_exp.apply_to_instance(x_raw) > 0) != ai_pred:
                for k in out: out[k]['mean_delta'] *= -1
        elif strat == "recall_change_dt":
            out = recall_change_dt(x_raw, mem_dt, bounds=bounds, k=3)
            t = out.pop('expected_time', 0.0); mem_dt.tick(t)
        else:
            direction = 'increase' if ai_pred==0 else 'decrease'
            out = recall_change_lr(mem_lr, k=6, preferred_direction=direction)
            t = out.pop('expected_time', 0.0); mem_lr.tick(t)

        out = smooth_probs_with_lapse(out, lapse=lapse, num_features=6)

        # model's edit for participant-chosen feature (for MAE)
        # print("feat chosen:", feat_chosen, " delta_chosen:", delta_chosen)
        try:
            int(float(feat_chosen[1:]))
        except:
            continue
        f_idx = int(float(feat_chosen[1:])) if isinstance(feat_chosen,str) and feat_chosen.startswith('a') else int(feat_chosen)
        feat_chosen = f'a{f_idx}'
        edited = apply_change_to_feature(x_raw, feat_chosen, bounds, out[feat_chosen]['mean_delta'],
                                         over_margin=over_margin)
        delta_pred = edited[f_idx] - x_raw[f_idx]

        # NLL & MAE
        nll = _nll_for_choice(out, feat_chosen, lapse=lapse, num_features=6)
        # if nll>3:
        #     print(out, feat_chosen, iid, app_id, condition, shown, strategies[strat_id])
        mae = abs(delta_pred - delta_chosen)
        mae /= (bounds[f'a{f_idx}'][1] - bounds[f'a{f_idx}'][0])  # normalize by range

        # sample model’s own feature/delta and compute flips
        feat_samp, delta_samp = sample_from_probs(out)   # returns (feat, delta)
        x_samp = apply_change_to_feature(x_raw, feat_samp, bounds, delta_samp, over_margin=over_margin)
        ai_cf  = run_ai_prediction(x_samp, transform, ai)
        if shown=='DT':
            xai_cf = int(dt_exp.apply_to_instance(x_samp)['class_index'])
        else:
            xai_cf = int(lr_exp.apply_to_instance(x_samp) > 0)

        counts[strat_id]+=1
        mtime[strat_id] = (mtime[strat_id]*(counts[strat_id]-1)+t)/counts[strat_id]
        step += 1

        NLLs.append(nll); MAEs.append(mae); TIMES.append(t)

        # ----- per-trial log row -----

        row = {
            **{k: (v.item() if hasattr(v, "item") else v) for k, v in tr.to_dict().items()},  # trial cols first
            **{k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()},  # participant info next
            "Participant Id": participant_id,                # override/add if not already in tr
            "Trial Index": int(ti),

            # model-sampled edit summary
            "Model strategy": strat,
            "Model depth": int(depth) if strat == "change_path_dt" else None,
            "Model changed feature index": feat_samp[1:] if isinstance(feat_samp,str) and feat_samp.startswith('a') else int(feat_samp),
            "Model changed feature name": str(feat_samp),
            "Model changed amount": float(delta_samp),
            "Model AI prediction (CF)": int(ai_cf),
            "Model changed AI prediction": int(ai_cf != ai_pred),
            "Model XAI prediction (CF)": int(xai_cf),
            "Model changed XAI prediction": int(xai_cf != ai_pred),  # if you meant vs XAI, compare to xai_orig instead

            # per-participant-feature comparison
            "Model mean_delta for chosen feature": float(out.get(feat_chosen, {}).get("mean_delta", 0.0)),
            "Trial NLL": float(nll),
            "Trial MAE": float(mae),

            # hyperparams
            "retrieval_threshold": float(rt),
            "over_margin": float(over_margin),
            "chi": float(chi),
        }
        trials_log.append(row)
    return dict(
        nll=float(np.mean(NLLs)) if NLLs else 1e3,
        mae=float(np.mean(MAEs)) if MAEs else 1e3,
        time=float(np.mean(TIMES)) if TIMES else 0.0,
        trials=trials_log
    )

def fit_participant_with_gpbo(model, user_loader, participant_id, ai_dataset_loader,
                              lr_df, dt_df, metadata_df, n_init=8, n_iter=30, lapse=0.1,
                              bounds=dict(retrieval_threshold=(-2.0,0.5), over_margin=(0.0,0.5), chi=(0.0,0.02)),
                              alpha_mae=1.0, beta_time=0.0, seed=0):
    rng = np.random.default_rng(seed)
    def obj(theta):
        lap = theta.get('lapse', lapse)
        s = score_participant_with_theta(model, user_loader, participant_id, ai_dataset_loader,
                                         lr_df, dt_df, metadata_df, theta, lapse=lap)
        score = s['nll'] + alpha_mae*s['mae'] + beta_time*(theta['chi']*s['time'])
        return score, s

    X, y = [], []
    best = {'obj': np.inf}; best_trials = []
    def sample_theta():
        return dict(
            retrieval_threshold=rng.uniform(*bounds['retrieval_threshold']),
            over_margin=rng.uniform(*bounds['over_margin']),
            chi=rng.uniform(*bounds['chi']),
            lapse=rng.uniform(*bounds['lapse'])
        )

    # init
    for _ in range(n_init):
        th = sample_theta(); val, s = obj(th)
        X.append([th['retrieval_threshold'], th['over_margin'], th['chi'], th['lapse']]); y.append(val)
        if val < best['obj']:
            best = {**th, 'obj': float(val), 'nll': s['nll'], 'mae': s['mae'], 'time': s['time']}
            best_trials = s['trials']
    X, y = np.array(X), np.array(y)

    gp = GaussianProcessRegressor(
        kernel=C(1.0,(1e-3,1e3)) * Matern(length_scale=[0.5,0.2,0.01,0.05], nu=2.5) + WhiteKernel(1e-4,(1e-8,1e-1)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=3, random_state=seed
    ).fit(X,y)

    for _ in range(n_iter):
        grid = np.stack([
            rng.uniform(*bounds['retrieval_threshold'], size=200),
            rng.uniform(*bounds['over_margin'], size=200),
            rng.uniform(*bounds['chi'], size=200),
            rng.uniform(*bounds['lapse'], size=200)
        ], axis=1)
        mu, sig = gp.predict(grid, return_std=True)
        x_next = grid[np.argmax(_ei(mu, sig, best=np.min(y), xi=0.01))]
        th = dict(retrieval_threshold=x_next[0], over_margin=x_next[1], chi=x_next[2], lapse=x_next[3])
        val, s = obj(th)
        X = np.vstack([X, x_next]); y = np.append(y, val); gp.fit(X,y)
        if val < best['obj']:
            best = {**th, 'obj': float(val), 'nll': s['nll'], 'mae': s['mae'], 'time': s['time']}
            best_trials = s['trials']

    # pack best + trials so you don't need to re-run
    return {**best, 'participant_id': participant_id, 'trials': best_trials, 'X': X, 'y': y}

# ---- Batch: many participants, save one CSV of all best-trial logs ----
def fit_many_and_save_csv(model, user_loader, participant_ids, ai_dataset_loader,
                          lr_df, dt_df, metadata_df, out_csv="rl_fit_trials.csv",
                          **gpbo_kwargs):
    all_rows, summaries = [], []
    i = 0
    for pid in participant_ids:
        i += 1
        print(f"Fitting participant {i}/{len(participant_ids)}: {pid}")
        res = fit_participant_with_gpbo(model, user_loader, pid, ai_dataset_loader,
                                        lr_df, dt_df, metadata_df, **gpbo_kwargs)
        # append trials with summary fields
        for r in res['trials']:
            all_rows.append({
                **r,
                'Best NLL': res['nll'],
                'Best MAE': res['mae'],
                'Best time': res['time'],
                'Best retrieval_threshold': res['retrieval_threshold'],
                'Best over_margin': res['over_margin'],
                'Best chi': res['chi'],
                'Participant Id': res['participant_id'],
            })
        summaries.append({k: res[k] for k in ('participant_id','nll','mae','time','retrieval_threshold','over_margin','chi','lapse','obj')})

    df_trials = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_trials.to_csv(out_csv, index=False)
    return df_trials, pd.DataFrame(summaries)

# -------- Example usage --------
participant_list = [p for p in user_loader.get_participant_ids()
                    if user_loader.get_participant_info(p)['condition'] in ('LR','DT','DT+LR')]
subset = random.sample(participant_list, k=min(50, len(participant_list)))

model = PPO.load("model_counterfactual/simple_chi_model.zip")
df_trials, df_summary = fit_many_and_save_csv(
    model=model,
    user_loader=user_loader,
    participant_ids=subset,
    ai_dataset_loader=ai_dataset_loader,
    lr_df=lr_df, dt_df=dt_df, metadata_df=metadata_df,
    out_csv="outputs/rl_fit_trials.csv",
    n_init=8, n_iter=24, lapse=0.1,
    bounds=dict(retrieval_threshold=(-2.0,0.5), over_margin=(0.2,0.5), chi=(0.0,0.02), lapse=(0, 1.0)),
    alpha_mae=2.0, beta_time=0.0, seed=123
)
print("Saved:", "outputs/rl_fit_trials.csv")
print(df_summary.head())
