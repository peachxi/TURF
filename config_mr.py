import torch, os, math

class ConfigMR:
    # ================ Reliability-weighted Feature Fusion ================
    # When enabled, reliability-aware modality weights are also applied to
    # feature-level fusion branch y_feat, not only to prediction-level fusion.
    use_reliability_feature_fusion = False
    rff_detach_weights = True
    rff_scale = 3.0

    # ================ Data / IO ================
    data_path = 'E:\\cmff-main\\datasets\\MOSI\\unaligned_50.pkl'
    dataset_name = 'MOSI'
    save_dir = './ckpt_mosi_maeonly'
    os.makedirs(save_dir, exist_ok=True)

    # 实验标签（消融/对比时会打在日志里）
    experiment_tag = 'full'

    # ================ Runtime ================
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ================ BERT backbone (24G) ================
    bert_model = 'bert-base-uncased'
    freeze_text = True
    unfreeze_last_n = 8           # 全解冻（base 12层）
    hidden_dim = 128
    use_layer_pool = True          # last-4 平均

    # ================ Token-level uncertainty ================
    use_token_uncertainty = True
    use_token_bilstm = True
    token_lstm_hidden = 128
    token_head_dropout = 0.10
    token_logvar_bound = True
    precision_detach = False
    precision_temp = 0.95

    # ================ Audio / Vision ================
    audio_dim = 74                 # 将在运行时覆盖成真实维度
    vision_dim = 47
    av_hidden_dim = 128
    use_av_bilstm = True
    av_lstm_hidden = 128
    av_downsample_stride = 5
    use_ltp_audio = True
    use_ltp_vision = True
    ltp_dropout = 0.10

    # ================ Fusion weights (alpha) ================
    alpha_pred = 0.50
    alpha_feat = 0.30
    alpha_seq  = 0.20
    alpha_pred_init = 0.50
    alpha_pred_target = 0.50
    alpha_feat_init = 0.30
    alpha_feat_target = 0.30
    alpha_seq_target = 0.20
    alpha_feat_min = 0.10
    normalize_alpha_weights = True
    alpha_ramp_epochs = 4

    # ================ Train ================
    batch_size = 4
    gradient_accumulation_steps = 4
    use_amp = True
    clip_grad_norm = 1.0
    max_epochs = 30
    patience = 8

    # ================ Optim LRs ================
    lr_head = 2e-3
    lr_gating = 2e-3
    lr_av = 2e-3
    lr_backbone = 2e-5
    weight_decay = 1e-4

    # ================ Warmdown near token start ================
    warmdown_on_token_start = True
    warmdown_factor = 0.5
    warmdown_epochs = 2

    # ================ EMA ================
    use_ema = True
    ema_decay = 0.997
    ema_start_epoch = 5

    # ================ Heteroscedastic reg ================
    logvar_min = -2.5
    logvar_max = 1.5
    # Separate bounds prevent A/V modality log-variance from saturating at the
    # token-level upper bound. Token/text uncertainty stays conservative;
    # audio/vision can express stronger degradation under corruption.
    token_logvar_min = -2.5
    token_logvar_max = 1.5
    text_logvar_min = -2.5
    text_logvar_max = 1.5
    modality_logvar_min = -2.5
    modality_logvar_max = 2.5
    logvar_target_low = -1.8
    logvar_target_high = -1.0
    w_var_reg = 0.02
    min_logvar_pull_weight = 0.015

    # ================ Loss ramps ================
    w_main = 1.0
    w_sent_hetero = 0.10
    w_token_hetero_target = 0.25
    # ============================================================
    # Unimodal auxiliary supervision
    # Optional direct supervision for unimodal prediction heads.
    # ============================================================
    use_unimodal_aux: bool = False
    aux_text_weight: float = 0.0
    aux_audio_weight: float = 0.0
    aux_vision_weight: float = 0.0
    aux_huber_beta: float = 0.5
    aux_start_epoch: int = 1
    aux_ramp_epochs: int = 1

    w_consistency_target = 0.03
    w_smooth_target = 0.01
    token_start_epoch = 4
    token_ramp_epochs = 5
    token_ramp_cosine = True

    # ================ Corr/CCC ================
    use_corr_loss = True
    corr_weight = 0.06
    use_ccc_loss = True
    ccc_weight = 0.03
    corr_adapt = True
    corr_floor_pcc = 0.80

    # ================ Acc/F1 强化 ================
    use_sign_aux = True
    sign_weight = 0.08             # 末期想再抬F1可到 0.10~0.12
    sign_temp = 1.2
    use_focal = True
    focal_gamma = 2.0
    focal_alpha = 0.30

    # ================ Reliability gating ================
    freeze_gate_epochs = 2
    gate_reliability_gamma_target = 0.45
    reliability_start_epoch = 10
    reliability_ramp_epochs = 4
    gate_temp = 1.15

    # ================ Entropy band ================
    entropy_target_low = 0.60
    entropy_target_high = 0.85
    entropy_band_penalty = 0.02
    entropy_floor = 0.45
    entropy_ceiling = 1.05
    entropy_clamp_penalty = 0.02

    # ================ Modal floors ================
    weight_floor_text  = 0.08
    weight_floor_audio = 0.02
    weight_floor_vision= 0.02
    modal_floor_text  = 0.08
    modal_floor_audio = 0.02
    modal_floor_vision= 0.02
    modal_floor_start_epoch = 1
    modal_floor_ramp_epochs = 3
    modal_floor_decay_epoch = 10
    # 1.0 keeps per-sample dynamic gates intact. Values below 1.0 mix each
    # sample gate with the running batch mean and should only be used as a
    # deliberate regularizer.
    modality_weight_smoothing = 1.0

    # ================ Reliability EMA / normalization ================
    use_r_vec_ema = True
    r_vec_ema_beta = 0.9
    use_r_vec_modal_norm = True



    # ================ Reliability-supervised uncertainty training ================
    # Clean-corrupt paired supervision: force the corrupted modality to become less reliable.
    use_reliability_rank = False
    rel_rank_weight = 0.02
    rel_rank_margin = 0.20
    rel_rank_start_epoch = 1
    corrupt_prob = 0.50

    # Corruption strengths used only during reliability-supervised training.
    mask_token_id = 103             # BERT [MASK] for bert-base-uncased
    text_mask_prob = 0.15
    audio_noise_std = 0.50
    audio_frame_drop_prob = 0.25
    visual_dropout_prob = 0.30

    # V2: sometimes replace the selected modality with a full missing view.
    # This matches missing_audio / missing_vision stress tests better than frame-level corruption only.
    full_modality_drop_prob = 0.25

    # Optional gate supervision: corrupted modality should receive a lower dynamic gate weight.
    use_gate_suppression = False
    gate_supp_weight = 0.01
    gate_supp_margin = 0.03

    # V2: direct clean-vs-corrupt quality supervision on modality log-uncertainty.
    # Clean modality should have larger quality logit (-s); corrupted modality should have lower quality.
    use_quality_bce = False
    quality_bce_weight = 0.02

    # Optional uncertainty-dominant fusion branch. Keep disabled in the first run.
    use_uncertainty_dominant_fusion = False
    uncertainty_fusion_beta = 0.80
    use_fused_nll = False
    fused_nll_weight = 0.05

    # ================ Contrast & PosCorr ================
    use_contrast_loss = True
    contrast_weight = 0.02
    contrast_margin = 0.0
    use_poscorr_loss = True
    poscorr_weight = 0.02

    # ================ Spread ================
    use_spread_loss = True
    spread_target_ratio = 0.95
    w_spread = 0.01

    # ================ Composite early stop（含F1奖励） ================
    lambda_mae_corr = 0.15
    lambda_mae_ccc  = 0.10
    lambda_f1_bonus = 0.08

    # ================ Diagnostics / Threshold scan ================
    print_freq = 1
    diag_csv = os.path.join(save_dir, 'diag.csv')
    thresh_scan_min = -0.35
    thresh_scan_max = 0.35
    thresh_scan_steps = 71

    # ================ Sign Head 校准（评估期网格搜索） ================
    cls_calib_enable = True
    cls_calib_acc_floor = 0.83
    cls_calib_T_min = 0.6
    cls_calib_T_max = 1.6
    cls_calib_T_steps = 21
    cls_calib_thr_min = -0.4
    cls_calib_thr_max = 0.4
    cls_calib_thr_steps = 81
    select_best_by = 'mae'
