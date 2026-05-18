"""
trainer_mr.py - Complete Version with Visualization Support
Multi-modal Sentiment Analysis Trainer with Uncertainty Modeling
"""

import os, time, csv, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from model_mr import EMAWeights

# ---------------- Utility & Metrics ----------------
def set_seed(seed:int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _spearman(x,y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x,y).correlation)
    except Exception:
        x=x.argsort().argsort().astype(float); y=y.argsort().argsort().astype(float)
        if x.std()<1e-8 or y.std()<1e-8: return 0.0
        return float(np.corrcoef(x,y)[0,1])

def compute_all_metrics(preds,labels,thresh=0.0):
    y=labels.astype(float); yhat=preds.astype(float)
    mae=float(np.mean(np.abs(yhat-y)))
    pcc=0.0 if (yhat.std()<1e-8 or y.std()<1e-8) else float(np.corrcoef(yhat,y)[0,1])
    spr=_spearman(yhat.copy(),y.copy())
    mx,my=yhat.mean(),y.mean(); vx,vy=yhat.var(),y.var()
    cov=float(np.mean((yhat-mx)*(y-my)))
    ccc=(2*cov)/(vx+vy+(mx-my)**2 + 1e-8)
    yi=np.clip(np.rint(y),-3,3).astype(int); yh=np.clip(np.rint(yhat),-3,3).astype(int)
    acc7=float(np.mean(yi==yh))
    def bin_set(mask):
        yy,pp=y[mask],yhat[mask]
        if yy.size==0: return 0.0,0.0,0.0
        gt=yy>thresh; pr=pp>thresh
        acc=float(np.mean(pr==gt))
        tp=np.sum(pr & gt); fp=np.sum(pr & (~gt)); fn=np.sum((~pr)&gt)
        prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
        f1=2*prec*rec/(prec+rec+1e-8)
        return acc,prec,f1
    acc2_all,_,f1_all=bin_set(np.ones_like(y,dtype=bool))
    acc2_nz,_,f1_nz=bin_set(y!=0)
    return {'mae':mae,'pcc':pcc,'spr':spr,'ccc':ccc,'acc7':acc7,
            'acc2_all':acc2_all,'f1_all':f1_all,'acc2_nonzero':acc2_nz,'f1_nonzero':f1_nz}

def scan_best_thresh(preds,labels,lo=-0.35,hi=0.35,steps=71,objective='f1_nonzero'):
    cands=np.linspace(lo,hi,steps)
    best_t,best_val=0.0,-1.0
    for t in cands:
        m=compute_all_metrics(preds,labels,thresh=t)
        v=m.get(objective,0.0)
        if v>best_val:
            best_val=v; best_t=float(t)
    return best_t

def _cls_metrics_from_logits(logits,labels,T=1.0,thr=0.0,nonzero=True):
    y=labels.astype(float)
    mask=(y!=0) if nonzero else np.ones_like(y,dtype=bool)
    if mask.sum()==0: return 0.0,0.0
    z=logits[mask]/max(T,1e-6)
    gt=(y[mask]>0); pr=(z>thr)
    acc=float((pr==gt).mean())
    tp=np.sum(pr & gt); fp=np.sum(pr & (~gt)); fn=np.sum((~pr)&gt)
    prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
    f1=2*prec*rec/(prec+rec+1e-8)
    return acc,f1

def _grid_search_sign(logits,labels,acc_floor=0.83,T_min=0.6,T_max=1.6,T_steps=21,
                      thr_min=-0.4,thr_max=0.4,thr_steps=81):
    Ts=np.linspace(T_min,T_max,T_steps); thrs=np.linspace(thr_min,thr_max,thr_steps)
    best={'T':1.0,'thr':0.0,'acc':0.0,'f1':0.0}
    for T in Ts:
        for thr in thrs:
            acc,f1=_cls_metrics_from_logits(logits,labels,T=T,thr=thr,nonzero=True)
            if acc>=acc_floor and f1>best['f1']:
                best={'T':float(T),'thr':float(thr),'acc':acc,'f1':f1}
    return best

# ---------------- Loss helpers ----------------
def hetero_sentence_loss(mus,logvars,y):
    diff2=(mus - y.unsqueeze(1))**2
    return torch.mean(diff2*torch.exp(-logvars) + logvars)

def token_hetero_loss(mu_tokens,lv_tokens,y,mask):
    diff2=(mu_tokens - y.unsqueeze(1))**2
    precision=torch.exp(-lv_tokens)
    nll=(diff2*precision + lv_tokens)*mask
    return nll.sum()/mask.sum().clamp_min(1.0)

def token_smooth_loss(lv_tokens,mask):
    if lv_tokens.size(1)<2: return torch.zeros((),device=lv_tokens.device)
    diff=torch.abs(lv_tokens[:,1:] - lv_tokens[:,:-1])
    valid=mask[:,1:]*mask[:,:-1]
    return (diff*valid).sum()/valid.sum().clamp_min(1.0)

def corr_loss(yhat,y):
    yc=yhat - yhat.mean(); yt=y - y.mean()
    denom=(yc.pow(2).sum().sqrt()*yt.pow(2).sum().sqrt()+1e-8)
    p=(yc*yt).sum()/denom
    return 1.0-p

def ccc_loss(yhat,y):
    yhat_m,y_m=yhat.mean(),y.mean()
    yhat_v,y_v=yhat.var(unbiased=False),y.var(unbiased=False)
    cov=((yhat - yhat_m)*(y - y_m)).mean()
    ccc=2*cov/(yhat_v + y_v + (yhat_m - y_m)**2 + 1e-8)
    return 1.0-ccc

def var_boundary_reg(lv_tokens,low,high):
    below=F.relu(low - lv_tokens); above=F.relu(lv_tokens - high)
    return (below+above).mean()

def min_logvar_pull(lv_tokens,margin=-2.2):
    return F.relu(margin - lv_tokens).mean()

def focal_bce_with_logits(logits,target,alpha=0.25,gamma=2.0):
    logits=logits.float()
    target=target.float()
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    pt=torch.exp(-bce)
    w=alpha*target + (1-alpha)*(1-target)
    loss=w*((1-pt).pow(gamma))*bce
    return loss.mean()

# ---------------- Trainer ----------------
class HybridTrainer:
    def __init__(self,model,loaders,cfg):
        self.model=model
        self.train_loader,self.valid_loader,self.test_loader=loaders
        self.cfg=cfg
        self.device=cfg.device
        self.model.to(self.device)
        set_seed(getattr(cfg,'seed',3407))

        if not hasattr(self.cfg,'diag_csv') or not self.cfg.diag_csv:
            self.cfg.diag_csv=os.path.join(self.cfg.save_dir,'diag.csv')
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        self._ensure_modal_projs_before_optimizer()

        mode=str(getattr(cfg,'gating_mode','dynamic')).lower()
        hard_modes=['static_equal_hard','static_learned_hard']

        head_params=[]; gating_params=[]; av_params=[]; backbone_params=[]

        for n,p in self.model.named_parameters():
            if not p.requires_grad: continue
            if mode in hard_modes and (n.startswith('gating') or n=='static_gate_logits'):
                continue
            if n.startswith('gating') or n=='static_gate_logits':
                gating_params.append(p)
            elif n.startswith(('a_lstm','v_lstm','a_proj','v_proj','audio_in','vision_in')):
                av_params.append(p)
            elif n.startswith('bert.'):
                backbone_params.append(p)
            else:
                head_params.append(p)

        groups=[]
        if head_params: groups.append({'params':head_params,'lr':getattr(cfg,'lr_head',2e-3),'weight_decay':cfg.weight_decay})
        if gating_params: groups.append({'params':gating_params,'lr':getattr(cfg,'lr_gating',2e-3),'weight_decay':cfg.weight_decay})
        if av_params: groups.append({'params':av_params,'lr':getattr(cfg,'lr_av',2e-3),'weight_decay':cfg.weight_decay})
        if backbone_params: groups.append({'params':backbone_params,'lr':getattr(cfg,'lr_backbone',2e-5),'weight_decay':cfg.weight_decay})
        self.opt=torch.optim.AdamW(groups)

        self.use_amp=bool(getattr(cfg,'use_amp',True))
        self.scaler=torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.use_ema=bool(getattr(cfg,'use_ema',True))
        self.ema=EMAWeights(self.model,decay=getattr(cfg,'ema_decay',0.997)) if self.use_ema else None
        self.ema_active=False; self.ema_updates=0
        self.ema_min_updates_for_eval=int(getattr(cfg,'ema_min_updates_for_eval',200))

        # 初始化 CSV
        if not os.path.exists(cfg.diag_csv):
            with open(cfg.diag_csv,'w',newline='',encoding='utf-8') as f:
                csv.writer(f).writerow([
                    'epoch','split','mae','pcc','spr','ccc','acc7','acc2_all','f1_all','acc2_nz','f1_nz',
                    'pred_mean','pred_std','mae_base','mae_seq','mae_feat','parts_mae_std',
                    'mean_token_logvar','token_logvar_std','mean_token_precision','token_precision_std',
                    'corr_err_logvar','gate_entropy','entropy_band_pen','entropy_clamp_pen','kl_align',
                    'w_text_mean','w_audio_mean','w_vision_mean','w_text_std','w_audio_std','w_vision_std',
                    'w_batch_var_text','w_batch_var_audio','w_batch_var_vision',
                    'hard_gain','easy_pen','var_reg_loss','min_logvar_pull_loss','contrast_loss','poscorr_loss',
                    'grad_head','grad_gating','grad_backbone','grad_av','thresh_used','gating_mode','tag'
                ])

        self.best_val_composite=float('inf'); self.patience=0
        self._last_gn=(0.0,0.0,0.0,0.0)
        self._best_thresh=0.0
        self.prev_w_mean=None
        self._sign_calib=None
        self._joint_calib=None
        self.current_epoch=0
        alpha_pred_base = getattr(self.cfg, 'alpha_pred', 0.50)
        alpha_seq_base = getattr(self.cfg, 'alpha_seq', 0.20)
        alpha_feat_base = getattr(self.cfg, 'alpha_feat', 0.30)
        alpha_pred = getattr(self.cfg, 'alpha_pred_target', alpha_pred_base)
        alpha_feat = getattr(self.cfg, 'alpha_feat_target', alpha_feat_base)
        alpha_seq = getattr(self.cfg, 'alpha_seq_target', alpha_seq_base)
        self._set_alpha_triplet(alpha_pred, alpha_feat, alpha_seq)
        tau = 1.0
        self._wTok = tau * getattr(self.cfg, 'w_token_hetero_target', 0.25)
        self._wCon = tau * getattr(self.cfg, 'w_consistency_target', 0.03)
        self._wSm = tau * getattr(self.cfg, 'w_smooth_target', 0.01)
        self._gate_reliability_gamma_target = float(getattr(self.cfg, 'gate_reliability_gamma_target', 0.30))
        self.cfg.gate_reliability_gamma_current = 0.0
        self.progress_step_interval = int(getattr(cfg, 'progress_step_interval', 100))

    def _ensure_modal_projs_before_optimizer(self):
        if not hasattr(self.model, '_ensure_modal_projs'):
            return
        try:
            first_batch = next(iter(self.train_loader))
            a = first_batch['audio'].to(self.device)
            v = first_batch['vision'].to(self.device)
            self.cfg.audio_dim = int(a.shape[-1])
            self.cfg.vision_dim = int(v.shape[-1])
            self.model._ensure_modal_projs(a, v)
            self.model.to(self.device)
            print(f"[Init] modal projections ready before optimizer: audio_dim={self.cfg.audio_dim} vision_dim={self.cfg.vision_dim}")
        except Exception as exc:
            print(f"[Warn] could not initialize modal projections before optimizer: {exc}")

    def _accum(self): return max(1,int(getattr(self.cfg,'gradient_accumulation_steps',1)))

    def _cfg_float(self, name, legacy_name=None, default=0.0):
        if hasattr(self.cfg, name):
            return float(getattr(self.cfg, name))
        if legacy_name is not None and hasattr(self.cfg, legacy_name):
            return float(getattr(self.cfg, legacy_name))
        return float(default)

    def _modal_floor_vec(self, device):
        return torch.tensor([
            self._cfg_float('modal_floor_text', 'weight_floor_text', 0.0),
            self._cfg_float('modal_floor_audio', 'weight_floor_audio', 0.0),
            self._cfg_float('modal_floor_vision', 'weight_floor_vision', 0.0)
        ], device=device).unsqueeze(0)

    def _set_alpha_triplet(self, alpha_pred, alpha_feat, alpha_seq):
        if bool(getattr(self.cfg,'normalize_alpha_weights',True)):
            denom=max(float(alpha_pred)+float(alpha_feat)+float(alpha_seq),1e-8)
            alpha_pred,alpha_feat,alpha_seq=alpha_pred/denom,alpha_feat/denom,alpha_seq/denom
        self.alpha_triplet=(alpha_pred,alpha_feat,alpha_seq)
    
    def _cosine_ramp(self,t): return 0.5*(1.0 - math.cos(math.pi*max(0.0,min(1.0,t))))

    def _apply_warmdown_if_needed(self,epoch):
        if (getattr(self.cfg,'warmdown_on_token_start',True) and
            getattr(self.cfg,'token_start_epoch',4)<=epoch<
            getattr(self.cfg,'token_start_epoch',4)+getattr(self.cfg,'warmdown_epochs',2)):
            for g in self.opt.param_groups:
                g['lr']=g['lr']*getattr(self.cfg,'warmdown_factor',0.5)

    def _schedule_weights(self,epoch):
        t_start=getattr(self.cfg,'token_start_epoch',4)
        t_ramp=max(1,int(getattr(self.cfg,'token_ramp_epochs',5)))
        tau=0.0 if epoch<t_start else min(1.0,(epoch - t_start +1)/t_ramp)
        tau=self._cosine_ramp(tau) if bool(getattr(self.cfg,'token_ramp_cosine',True)) else tau
        self._wTok=tau*getattr(self.cfg,'w_token_hetero_target',0.25)
        self._wCon=tau*getattr(self.cfg,'w_consistency_target',0.03)
        self._wSm=tau*getattr(self.cfg,'w_smooth_target',0.01)

        a_ramp=max(1,int(getattr(self.cfg,'alpha_ramp_epochs',4)))
        alpha_pred_base=getattr(self.cfg,'alpha_pred',0.50)
        alpha_feat_base=getattr(self.cfg,'alpha_feat',0.30)
        alpha_seq_base=getattr(self.cfg,'alpha_seq',0.20)
        alpha_pred_init=getattr(self.cfg,'alpha_pred_init',alpha_pred_base)
        alpha_pred_target=getattr(self.cfg,'alpha_pred_target',alpha_pred_base)
        alpha_feat_init=getattr(self.cfg,'alpha_feat_init',alpha_feat_base)
        alpha_feat_target=getattr(self.cfg,'alpha_feat_target',alpha_feat_base)
        alpha_seq_target=getattr(self.cfg,'alpha_seq_target',alpha_seq_base)
        if epoch<=a_ramp:
            r=epoch/a_ramp
            alpha_pred=min(alpha_pred_target,
                           alpha_pred_init + (alpha_pred_target-alpha_pred_init)*r)
            alpha_feat=max(getattr(self.cfg,'alpha_feat_min',0.10),
                           alpha_feat_init + (alpha_feat_target-alpha_feat_init)*r)
            alpha_seq=min(alpha_seq_target,
                          alpha_seq_base + (alpha_seq_target-alpha_seq_base)*r)
        else:
            alpha_pred=alpha_pred_target
            alpha_feat=max(getattr(self.cfg,'alpha_feat_min',0.10), alpha_feat_target)
            alpha_seq=alpha_seq_target
        self._set_alpha_triplet(alpha_pred,alpha_feat,alpha_seq)

        mode=str(getattr(self.cfg,'gating_mode','dynamic')).lower()
        r_start=getattr(self.cfg,'reliability_start_epoch',9)
        r_ramp=max(1,int(getattr(self.cfg,'reliability_ramp_epochs',4)))
        if epoch<r_start or mode not in ['dynamic']:
            gamma=0.0
        else:
            tau_g=min(1.0,(epoch - r_start +1)/r_ramp)
            gamma=tau_g*self._gate_reliability_gamma_target
        self.cfg.gate_reliability_gamma_current=gamma

    def _grad_norms_now(self):
        head=gate=back=av=0.0
        for n,p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None: continue
            g=p.grad.detach().norm().item()
            if n.startswith('bert.'): back+=g
            elif n.startswith('gating') or n=='static_gate_logits': gate+=g
            elif n.startswith(('a_lstm','v_lstm','a_proj','v_proj','audio_in','vision_in')): av+=g
            else: head+=g
        self._last_gn=(head,gate,back,av)

    def _normalize_r_vec(self,r_vec_batch,update_ema=True):
        mean_modal=r_vec_batch.detach().mean(dim=0)
        base=mean_modal
        if bool(getattr(self.cfg,'use_r_vec_ema',True)):
            if self.model.training and update_ema:
                if not hasattr(self,'r_vec_ema') or self.r_vec_ema is None:
                    self.r_vec_ema=mean_modal
                else:
                    beta=float(getattr(self.cfg,'r_vec_ema_beta',0.9))
                    self.r_vec_ema=beta*self.r_vec_ema.detach()+(1-beta)*mean_modal
            if hasattr(self,'r_vec_ema') and self.r_vec_ema is not None:
                base=self.r_vec_ema.detach().to(r_vec_batch.device)
        if bool(getattr(self.cfg,'use_r_vec_modal_norm',True)):
            normed=r_vec_batch/(base.unsqueeze(0)+1e-8)
            normed=normed.clamp(0.25,4.0)
        else:
            normed=r_vec_batch
        return normed

    def _poscorr_loss(self,abs_err_t,mean_lv_t):
        ex=abs_err_t - abs_err_t.mean()
        lx=mean_lv_t - mean_lv_t.mean()
        num=(ex*lx).sum()
        den=(ex.pow(2).sum().sqrt()*lx.pow(2).sum().sqrt()+1e-8)
        corr=num/den
        return F.relu(-corr)


    def _reliability_weighted_y_feat(self, out, w_use):
        """
        Reliability-weighted feature fusion (RFF).

        The original y_feat branch consumes raw concatenated features
        [h_T, h_A, h_V], which can bypass reliability-aware gating.
        RFF reuses the already computed final modality weights w_use and
        feeds [scale*w_T*h_T, scale*w_A*h_A, scale*w_V*h_V] to y_feat_mlp.
        """
        if not bool(getattr(self.cfg, "use_reliability_feature_fusion", False)):
            return out["y_feat"]

        feats = out.get("fusion_features", None)
        if feats is None:
            return out["y_feat"]

        if feats.dim() != 2 or feats.size(1) % 3 != 0:
            return out["y_feat"]

        bsz, dim = feats.shape
        hdim = dim // 3
        feats_3 = feats.view(bsz, 3, hdim)

        weights = w_use
        if bool(getattr(self.cfg, "rff_detach_weights", True)):
            weights = weights.detach()

        scale = float(getattr(self.cfg, "rff_scale", 3.0))
        feats_rel = (feats_3 * (weights.unsqueeze(-1) * scale)).reshape(bsz, 3 * hdim)

        y_feat_rel = self.model.y_feat_mlp(
            torch.cat([feats_rel, out["mu_text_seq"]], dim=1)
        ).squeeze(1)
        return y_feat_rel

    def _composite_val(self,s):
        return s['mae'] + 0.12*(1-s['pcc']) + 0.08*(1-s['ccc']) - 0.00*s['f1_nz']

    def _joint_search(self,val_s):
        if not bool(getattr(self.cfg,'enable_joint_calib',False)): return None
        if val_s.get('sign_logits') is None: return None
        if self.current_epoch < int(getattr(self.cfg,'joint_start_epoch', max(1,self.cfg.max_epochs-3))):
            return None
        logits=val_s['sign_logits']; labels=val_s['labels']
        Ts=np.linspace(getattr(self.cfg,'joint_T_min',0.8), getattr(self.cfg,'joint_T_max',1.6),
                       int(getattr(self.cfg,'joint_T_steps',7)))
        cls_space=np.linspace(getattr(self.cfg,'joint_thr_cls_min',-0.5), getattr(self.cfg,'joint_thr_cls_max',-0.1),
                              int(getattr(self.cfg,'joint_thr_cls_steps',9)))
        reg_space=np.linspace(getattr(self.cfg,'joint_thr_reg_min',-0.30), getattr(self.cfg,'joint_thr_reg_max',0.05),
                              int(getattr(self.cfg,'joint_thr_reg_steps',9)))
        acc_floor=float(getattr(self.cfg,'joint_acc_floor',0.83))
        best={'T':1.0,'thr_cls':0.0,'thr_reg':0.0,'macroF1':0.0,'F1_nz':0.0,'acc2_nz':0.0}
        y=labels; nz=(y!=0)
        for T in Ts:
            scaled=logits/T
            for tc in cls_space:
                pr=(scaled>tc)[nz]; gt=(y[nz]>0)
                tp=np.sum(pr & gt); fp=np.sum(pr & (~gt)); fn=np.sum((~pr)&gt); tn=np.sum((~pr)&(~gt))
                acc_cls=(tp+tn)/max(1,(tp+tn+fp+fn))
                if acc_cls<acc_floor: continue
                prec_pos=tp/(tp+fp+1e-8); rec_pos=tp/(tp+fn+1e-8)
                f1_pos=2*prec_pos*rec_pos/(prec_pos+rec_pos+1e-8)
                prec_neg=tn/(tn+fn+1e-8); rec_neg=tn/(tn+fp+1e-8)
                f1_neg=2*prec_neg*rec_neg/(prec_neg+rec_neg+1e-8)
                macro=0.5*(f1_pos+f1_neg)
                for tr in reg_space:
                    m=compute_all_metrics(val_s['preds'],labels,thresh=tr)
                    if macro>best['macroF1']:
                        best={'T':float(T),'thr_cls':float(tc),'thr_reg':float(tr),
                              'macroF1':macro,'F1_nz':m['f1_nonzero'],'acc2_nz':m['acc2_nonzero']}
        return best




    def _weights_from_out_noema(self, out, mode, epoch):
        """Compute dynamic weights for auxiliary/corrupted forward passes without EMA update."""
        if mode=='static_equal_hard':
            w_content=torch.full((out['mus'].size(0),3),1/3,device=out['mus'].device,dtype=out['mus'].dtype)
        elif mode in ['static_learned_hard','static_learned']:
            w_content=out['weights_static']
        elif mode=='static_equal':
            w_content=torch.full_like(out['weights_content'],1/3)
        else:
            w_content=out['weights_content']
        gmask=out['gate_mask']
        w_content=w_content*gmask
        zero_rows=(w_content.sum(dim=1,keepdim=True)<=1e-8)
        if zero_rows.any():
            fallback=torch.tensor([[1.0,0.0,0.0]],device=w_content.device,dtype=w_content.dtype)
            w_content=torch.where(zero_rows,fallback,w_content)
        w_content=w_content/w_content.sum(dim=1,keepdim=True).clamp_min(1e-8)
        if mode=='dynamic':
            r_vec_normed=self._normalize_r_vec(out['r_vec'],update_ema=False)
            gamma=float(getattr(self.cfg,'gate_reliability_gamma_current',
                                getattr(self.cfg,'gate_reliability_gamma_target',0.0)))
            if gamma>0 and epoch>getattr(self.cfg,'freeze_gate_epochs',2):
                gate_logits=(torch.log(w_content.float().clamp_min(1e-6))+
                             gamma*torch.log(r_vec_normed.float().clamp_min(1e-6)))
                w_rew=torch.softmax(gate_logits,dim=1).to(w_content.dtype)
            else:
                w_rew=w_content
            floor_vec=self._modal_floor_vec(w_content.device)
            w=w_rew + floor_vec
            w=w/w.sum(dim=1,keepdim=True).clamp_min(1e-8)
            return w
        return w_content

    def _make_corrupted_inputs(self, ids, mask, seg, audio, vision, modality_idx):
        """Create one corrupted view of the current batch. modality_idx: 0=text, 1=audio, 2=vision.

        V2 change: with probability `full_modality_drop_prob`, the selected modality is fully zeroed.
        This gives the reliability head an explicit missing-modality training signal, matching stress tests.
        """
        ids_c=ids.clone(); mask_c=mask.clone(); seg_c=seg.clone()
        audio_c=audio.clone(); vision_c=vision.clone()

        full_drop_p=float(getattr(self.cfg,'full_modality_drop_prob',0.0))
        do_full_drop=(full_drop_p>0 and float(torch.rand((),device=ids.device).item())<full_drop_p)

        if modality_idx==0:
            if do_full_drop:
                ids_c.zero_(); mask_c.zero_(); seg_c.zero_()
            else:
                mask_token_id=int(getattr(self.cfg,'mask_token_id',103))
                p=float(getattr(self.cfg,'text_mask_prob',0.15))
                # Avoid corrupting [CLS], [SEP], padding for BERT-style token ids.
                valid=(mask_c>0) & (ids_c!=0) & (ids_c!=101) & (ids_c!=102)
                rand=torch.rand(ids_c.shape,device=ids_c.device)
                ids_c=torch.where(valid & (rand<p), torch.full_like(ids_c,mask_token_id), ids_c)
        elif modality_idx==1:
            if do_full_drop:
                audio_c.zero_()
            else:
                noise_std=float(getattr(self.cfg,'audio_noise_std',0.50))
                drop_p=float(getattr(self.cfg,'audio_frame_drop_prob',0.25))
                scale=audio_c.float().std(dim=(1,2),keepdim=True).clamp_min(1e-4).to(audio_c.dtype)
                if noise_std>0:
                    audio_c=audio_c + noise_std*scale*torch.randn_like(audio_c)
                if drop_p>0:
                    drop=(torch.rand(audio_c.shape[:2],device=audio_c.device).unsqueeze(-1)<drop_p)
                    audio_c=audio_c.masked_fill(drop,0.0)
        elif modality_idx==2:
            if do_full_drop:
                vision_c.zero_()
            else:
                drop_p=float(getattr(self.cfg,'visual_dropout_prob',0.30))
                if drop_p>0:
                    drop=(torch.rand(vision_c.shape[:2],device=vision_c.device).unsqueeze(-1)<drop_p)
                    vision_c=vision_c.masked_fill(drop,0.0)
        return ids_c,mask_c,seg_c,audio_c,vision_c

    def _modality_log_uncertainty(self, out, modality_idx):
        """Return scalar log-uncertainty for the selected modality for each sample."""
        if modality_idx==0 and bool(getattr(self.cfg,'use_token_uncertainty',True)):
            return ((out['logvar_tokens']*out['mask']).sum(dim=1)/out['mask'].sum(dim=1).clamp_min(1.0))
        return out['logvars'][:,modality_idx]

    def _reliability_rank_aux_loss(self, out_clean, out_corrupt, modality_idx):
        # Make the corrupted view less reliable, i.e., larger log-variance.
        s_clean=self._modality_log_uncertainty(out_clean,modality_idx).detach()
        s_corrupt=self._modality_log_uncertainty(out_corrupt,modality_idx)
        margin=float(getattr(self.cfg,'rel_rank_margin',0.20))
        return F.relu(margin + s_clean - s_corrupt).mean()

    def _gate_suppression_aux_loss(self, w_clean, w_corrupt, modality_idx):
        # The corrupted modality should receive a lower gate weight than in the clean view.
        wc=w_clean[:,modality_idx].detach()
        wz=w_corrupt[:,modality_idx]
        margin=float(getattr(self.cfg,'gate_supp_margin',0.03))
        return F.relu(margin + wz - wc).mean()

    def _quality_bce_aux_loss(self, out_clean, out_corrupt, modality_idx):
        """Direct quality supervision for reliability.

        We use -s as a quality logit: clean modality -> high quality; corrupted modality -> low quality.
        This is stronger than pairwise ranking and is intended mainly for weak A/V reliability heads.
        """
        s_clean=self._modality_log_uncertainty(out_clean,modality_idx)
        s_corrupt=self._modality_log_uncertainty(out_corrupt,modality_idx)
        q_clean=(-s_clean).float()
        q_corrupt=(-s_corrupt).float()
        loss_clean=F.binary_cross_entropy_with_logits(q_clean, torch.ones_like(q_clean))
        loss_corrupt=F.binary_cross_entropy_with_logits(q_corrupt, torch.zeros_like(q_corrupt))
        return 0.5*(loss_clean+loss_corrupt)



    def _get_out_tensor(self, out, names):
        for name in names:
            if isinstance(out, dict) and name in out and out[name] is not None:
                value = out[name]
                if hasattr(value, "dim") and value.dim() > 1:
                    value = value.squeeze(-1)
                return value.float()
        return None

    def _unimodal_aux_loss(self, out, y, epoch):
        if not bool(getattr(self.cfg, "use_unimodal_aux", False)):
            return y.new_tensor(0.0), {}

        start_epoch = int(getattr(self.cfg, "aux_start_epoch", 1))
        ramp_epochs = max(1, int(getattr(self.cfg, "aux_ramp_epochs", 1)))
        if epoch < start_epoch:
            return y.new_tensor(0.0), {}

        tau = min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))
        y = y.float().view(-1)

        mu_text = self._get_out_tensor(out, ["mu_text", "mu_t", "y_text", "pred_text", "text_mu"])
        mu_audio = self._get_out_tensor(out, ["mu_audio", "mu_a", "y_audio", "pred_audio", "audio_mu"])
        mu_vision = self._get_out_tensor(
            out,
            [
                "mu_vision", "mu_visual", "mu_v",
                "y_vision", "y_visual",
                "pred_vision", "pred_visual",
                "vision_mu", "visual_mu",
            ],
        )

        beta = float(getattr(self.cfg, "aux_huber_beta", 0.5))
        wt = float(getattr(self.cfg, "aux_text_weight", 0.0)) * tau
        wa = float(getattr(self.cfg, "aux_audio_weight", 0.0)) * tau
        wv = float(getattr(self.cfg, "aux_vision_weight", 0.0)) * tau

        loss = y.new_tensor(0.0)
        logs = {}

        if mu_text is not None and wt > 0:
            lt = F.smooth_l1_loss(mu_text.view(-1), y, beta=beta)
            loss = loss + wt * lt
            logs["aux_text"] = float(lt.detach().cpu())

        if mu_audio is not None and wa > 0:
            la = F.smooth_l1_loss(mu_audio.view(-1), y, beta=beta)
            loss = loss + wa * la
            logs["aux_audio"] = float(la.detach().cpu())

        if mu_vision is not None and wv > 0:
            lv = F.smooth_l1_loss(mu_vision.view(-1), y, beta=beta)
            loss = loss + wv * lv
            logs["aux_vision"] = float(lv.detach().cpu())

        logs["aux_total"] = float(loss.detach().cpu())
        logs["aux_tau"] = tau
        return loss, logs

    def _run_epoch(self,loader,epoch,is_train=True,collect_preds=False):
        self.model.train() if is_train else self.model.eval()
        self.prev_w_mean=None
        mode=str(getattr(self.cfg,'gating_mode','dynamic')).lower()
        hard_modes=['static_equal_hard','static_learned_hard']
        preds=[]; labels=[]; sign_logits=[]
        tok_lv_means=[]; tok_prec_means=[]
        weight_rows=[]
        gate_entropy_list=[]; kl_list=[]
        mae_base_list=[]; mae_seq_list=[]; mae_feat_list=[]
        var_reg_list=[]; min_pull_list=[]; contrast_list=[]; poscorr_list=[]
        hard_gain_list=[]; easy_pen_list=[]

        total_loss=0.0; step=0
        accum=self._accum()
        main_loss_fn=nn.SmoothL1Loss(beta=0.6)
        start_time=time.time()

        for i,b in enumerate(loader,start=1):
            ids=b['input_ids'].to(self.device)
            mask=b['attention_mask'].to(self.device)
            seg=b['token_type_ids'].to(self.device)
            a=b['audio'].to(self.device); v=b['vision'].to(self.device)
            y=b['label'].to(self.device).float().view(-1)

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out=self.model(ids,mask,seg,a,v,epoch=epoch)
                    mus,logvars=out['mus'],out['logvars']

                    if mode=='static_equal_hard':
                        w_content=torch.full((ids.size(0),3),1/3,device=self.device)
                    elif mode=='static_learned_hard':
                        w_content=out['weights_static']
                    elif mode=='static_equal':
                        w_content=torch.full_like(out['weights_content'],1/3)
                    elif mode=='static_learned':
                        w_content=out['weights_static']
                    else:
                        w_content=out['weights_content']

                    gmask=out['gate_mask']
                    w_content=w_content*gmask
                    zero_rows=(w_content.sum(dim=1,keepdim=True)<=1e-8)
                    if zero_rows.any():
                        fallback=torch.tensor([[1.0,0.0,0.0]],device=w_content.device)
                        w_content=torch.where(zero_rows,fallback,w_content)
                    w_content=w_content/w_content.sum(dim=1,keepdim=True).clamp_min(1e-8)

                    if mode=='dynamic':
                        r_vec_normed=self._normalize_r_vec(out['r_vec'],update_ema=is_train)
                        gamma=float(getattr(self.cfg,'gate_reliability_gamma_current',
                                            getattr(self.cfg,'gate_reliability_gamma_target',0.0)))
                        if gamma>0 and epoch>getattr(self.cfg,'freeze_gate_epochs',2):
                            gate_logits=(torch.log(w_content.float().clamp_min(1e-6))+
                                         gamma*torch.log(r_vec_normed.float().clamp_min(1e-6)))
                            w_rew=torch.softmax(gate_logits,dim=1).to(w_content.dtype)
                        else:
                            w_rew=w_content
                        floor_vec=self._modal_floor_vec(w_content.device)
                        w_floor=w_rew + floor_vec
                        w_final=w_floor/w_floor.sum(dim=1,keepdim=True)
                        beta_w=float(getattr(self.cfg,'modality_weight_smoothing',1.0))
                        beta_w=max(0.0,min(1.0,beta_w))
                        if beta_w>=0.999:
                            w_use=w_final
                            self.prev_w_mean=w_final.detach().mean(dim=0,keepdim=True)
                        else:
                            if self.prev_w_mean is None:
                                self.prev_w_mean=w_final.detach().mean(dim=0,keepdim=True)
                            w_use=beta_w*w_final + (1-beta_w)*self.prev_w_mean.expand_as(w_final)
                            self.prev_w_mean=0.9*self.prev_w_mean + 0.1*w_final.detach().mean(dim=0,keepdim=True)
                    else:
                        w_use=w_content
                        if self.prev_w_mean is None:
                            self.prev_w_mean=w_use.detach().mean(dim=0,keepdim=True)

                    alpha_pred,alpha_feat,alpha_seq=self.alpha_triplet
                    fused_nll_var=None
                    if bool(getattr(self.cfg,'use_uncertainty_dominant_fusion',False)):
                        # Precision-style branch: uncertainty controls the main prediction path.
                        r_for_unc=self._normalize_r_vec(out['r_vec'],update_ema=False).clamp_min(1e-6)
                        tau=(w_content.detach()*r_for_unc).clamp_min(1e-6)
                        tau_sum=tau.sum(dim=1).clamp_min(1e-6)
                        y_unc=(tau*mus).sum(dim=1)/tau_sum
                        beta_unc=float(getattr(self.cfg,'uncertainty_fusion_beta',0.80))
                        beta_unc=max(0.0,min(1.0,beta_unc))
                        y_hat=beta_unc*y_unc + (1.0-beta_unc)*out['y_feat']
                        fused_nll_var=(1.0/tau_sum).clamp_min(1e-6)
                    else:
                        y_hat=(alpha_pred*(w_use*mus).sum(dim=1) +
                               alpha_feat * (self._reliability_weighted_y_feat(out, w_use) if bool(getattr(self.cfg, "use_reliability_feature_fusion", False)) else out["y_feat"]) +
                               alpha_seq*out['mu_text_seq'].squeeze(1))

                    loss=1.0*main_loss_fn(y_hat,y)
                    loss+=0.10*hetero_sentence_loss(mus,logvars,y)
                    if bool(getattr(self.cfg,'use_fused_nll',False)) and fused_nll_var is not None:
                        loss_fused_nll=0.5*(torch.log(fused_nll_var)+(y_hat-y).pow(2)/fused_nll_var).mean()
                        loss+=getattr(self.cfg,'fused_nll_weight',0.05)*loss_fused_nll

                    var_reg_val=torch.zeros((),device=y.device)
                    min_pull_val=torch.zeros((),device=y.device)
                    if bool(getattr(self.cfg,'use_token_uncertainty',True)):
                        loss+=self._wTok*token_hetero_loss(out['mu_tokens'],out['logvar_tokens'],y,out['mask'])
                        loss+=self._wCon*F.l1_loss(out['mu_text_seq'].squeeze(1),mus[:,0])
                        loss+=self._wSm*token_smooth_loss(out['logvar_tokens'],out['mask'])
                        if epoch>=getattr(self.cfg,'token_start_epoch',4) and float(getattr(self.cfg,'w_var_reg',0.0))>0:
                            var_reg_val=var_boundary_reg(out['logvar_tokens'],
                                                         getattr(self.cfg,'logvar_target_low',-1.8),
                                                         getattr(self.cfg,'logvar_target_high',-0.95))
                            min_pull_val=min_logvar_pull(out['logvar_tokens'],margin=-2.2)
                            loss+=getattr(self.cfg,'w_var_reg',0.0)*var_reg_val + \
                                   getattr(self.cfg,'min_logvar_pull_weight',0.0)*min_pull_val

                    if bool(getattr(self.cfg,'use_corr_loss',True)):
                        loss+=getattr(self.cfg,'corr_weight',0.06)*corr_loss(y_hat,y)
                    if bool(getattr(self.cfg,'use_ccc_loss',True)):
                        loss+=getattr(self.cfg,'ccc_weight',0.03)*ccc_loss(y_hat,y)

                    if bool(getattr(self.cfg,'use_sign_aux',True)):
                        nz=(y!=0)
                        if nz.any():
                            target=(y[nz]>0).float()
                            sign_logits_nz=getattr(self.cfg,'sign_temp',1.2)*out['sign_logit'][nz]
                            if bool(getattr(self.cfg,'use_focal',True)):
                                loss+=getattr(self.cfg,'sign_weight',0.08)*focal_bce_with_logits(
                                    sign_logits_nz,
                                    target,
                                    alpha=getattr(self.cfg,'focal_alpha',0.30),
                                    gamma=getattr(self.cfg,'focal_gamma',2.0))
                            else:
                                pos=max(int((target==1).sum().item()),1)
                                neg=max(int((target==0).sum().item()),1)
                                pos_weight=y.new_tensor(neg/pos)
                                loss+=getattr(self.cfg,'sign_weight',0.08)*F.binary_cross_entropy_with_logits(
                                    sign_logits_nz,target,pos_weight=pos_weight)

                    hard_gain=torch.zeros((),device=y.device)
                    easy_pen=torch.zeros((),device=y.device)
                    if mode=='dynamic' and bool(getattr(self.cfg,'use_hard_gain',True)):
                        y_eq=mus.mean(dim=1)
                        err_eq=(y_eq - y).abs().detach()
                        q_thr=torch.quantile(err_eq, getattr(self.cfg,'hard_gain_q',0.70))
                        hard_mask=(err_eq>q_thr).float()
                        easy_mask=1.0 - hard_mask

                        improve=(err_eq - (y_hat - y).abs())
                        hard_gain=F.relu(-improve)*hard_mask
                        gate_dev=(w_use - w_use.new_tensor([1/3,1/3,1/3])).abs().sum(dim=1)
                        easy_pen=F.relu(gate_dev - getattr(self.cfg,'max_gate_deviation_easy',0.06))*easy_mask
                        loss+=getattr(self.cfg,'hard_gain_weight',0.05)*hard_gain.mean()
                        loss+=0.02*easy_pen.mean()

                    gate_entropy=(-(w_use*torch.log(w_use.clamp_min(1e-8))).sum(dim=1)).mean()
                    gate_entropy_list.append(float(gate_entropy.item()))

                    r_vec_normed=self._normalize_r_vec(out['r_vec'],update_ema=False)
                    r_norm=r_vec_normed/r_vec_normed.sum(dim=1,keepdim=True).clamp_min(1e-8)
                    kl=(w_use.float()*(torch.log(w_use.float().clamp_min(1e-6))-torch.log(r_norm.float().clamp_min(1e-6)))).sum(dim=1)
                    kl_list.extend(kl.detach().cpu().numpy().tolist())

                    contrast_val=torch.zeros((),device=y.device)
                    if bool(getattr(self.cfg,'use_contrast_loss',False)) and mode=='dynamic':
                        u_modal=out.get('u_modal',None)
                        if u_modal is not None:
                            contrast_raw=(w_use*u_modal).sum(dim=1).mean()
                            contrast_val=F.relu(contrast_raw - getattr(self.cfg,'contrast_margin',0.0))
                            loss+=getattr(self.cfg,'contrast_weight',0.02)*contrast_val

                    poscorr_val=torch.zeros((),device=y.device)
                    if bool(getattr(self.cfg,'use_poscorr_loss',True)) and bool(getattr(self.cfg,'use_token_uncertainty',True)):
                        tok_mean_lv_t=((out['logvar_tokens']*out['mask']).sum(dim=1)/out['mask'].sum(dim=1).clamp_min(1.0))
                        abs_err_t=(y_hat - y).abs().detach()
                        poscorr_val=self._poscorr_loss(abs_err_t,tok_mean_lv_t)
                        loss+=getattr(self.cfg,'poscorr_weight',0.02)*poscorr_val



                    # ---------------- Reliability-supervised clean-corrupt auxiliary training ----------------
                    if (is_train and bool(getattr(self.cfg,'use_reliability_rank',False)) and
                        epoch>=int(getattr(self.cfg,'rel_rank_start_epoch',1)) and
                        float(getattr(self.cfg,'rel_rank_weight',0.0))>0 and
                        float(torch.rand((),device=self.device).item())<float(getattr(self.cfg,'corrupt_prob',0.50))):
                        modality_idx=int(torch.randint(0,3,(1,),device=self.device).item())
                        ids_c,mask_c,seg_c,a_c,v_c=self._make_corrupted_inputs(ids,mask,seg,a,v,modality_idx)
                        out_c=self.model(ids_c,mask_c,seg_c,a_c,v_c,epoch=epoch)
                        loss_rel=self._reliability_rank_aux_loss(out,out_c,modality_idx)
                        loss+=getattr(self.cfg,'rel_rank_weight',0.02)*loss_rel
                        if bool(getattr(self.cfg,'use_gate_suppression',False)) and float(getattr(self.cfg,'gate_supp_weight',0.0))>0:
                            w_c=self._weights_from_out_noema(out_c,mode,epoch)
                            loss_gate=self._gate_suppression_aux_loss(w_use,w_c,modality_idx)
                            loss+=getattr(self.cfg,'gate_supp_weight',0.01)*loss_gate
                        if bool(getattr(self.cfg,'use_quality_bce',False)) and float(getattr(self.cfg,'quality_bce_weight',0.0))>0:
                            loss_quality=self._quality_bce_aux_loss(out,out_c,modality_idx)
                            loss+=getattr(self.cfg,'quality_bce_weight',0.02)*loss_quality

                    loss=loss/accum

            if is_train:
                # Optional unimodal auxiliary supervision.
                aux_loss, aux_logs = self._unimodal_aux_loss(out, y, epoch)
                loss = loss + aux_loss
                self.scaler.scale(loss).backward()
                if (i%accum==0) and (i//accum==1):
                    self._grad_norms_now()
                if i%accum==0:
                    if getattr(self.cfg,'clip_grad_norm',0.0)>0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       getattr(self.cfg,'clip_grad_norm',0.0))
                    self.scaler.step(self.opt); self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    if self.use_ema and self.ema_active:
                        self.ema.update(self.model); self.ema_updates+=1

            total_loss+=loss.item()*accum; step+=1
            preds.append(y_hat.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            weight_rows.append(w_use.detach().cpu().numpy())
            if collect_preds:
                sign_logits.append(out['sign_logit'].detach().cpu().numpy())

            lv_tok=out['logvar_tokens'].detach(); m=out['mask'].detach()
            tok_mean_lv_t=((lv_tok*m).sum(dim=1)/m.sum(dim=1).clamp_min(1.0))
            tok_mean_prec_t=((torch.exp(-lv_tok)*m).sum(dim=1)/m.sum(dim=1).clamp_min(1.0))
            tok_lv_means.extend(tok_mean_lv_t.cpu().numpy().tolist())
            tok_prec_means.extend(tok_mean_prec_t.cpu().numpy().tolist())

            mae_base_list.append(((w_use*mus).sum(dim=1)-y).abs().mean().item())
            mae_seq_list.append((out['mu_text_seq'].squeeze(1)-y).abs().mean().item())
            mae_feat_list.append((out['y_feat']-y).abs().mean().item())

            var_reg_list.append(float(var_reg_val.item()))
            min_pull_list.append(float(min_pull_val.item()))
            contrast_list.append(float(contrast_val.item()))
            poscorr_list.append(float(poscorr_val.item()))
            hard_gain_list.append(float(hard_gain.mean().item()))
            easy_pen_list.append(float(easy_pen.mean().item()))

            if is_train and i % self.progress_step_interval == 0:
                elapsed=time.time()-start_time
                avg_step=elapsed/i
                eta=avg_step*(len(loader)-i)
                print(f"[TrainProg] ep={epoch} step={i}/{len(loader)} "
                      f"loss={loss.item()*accum:.4f} avg={avg_step:.3f}s ETA={eta/60:.1f}m mode={mode}")

        preds_arr=np.concatenate(preds,0); labels_arr=np.concatenate(labels,0)
        metrics=compute_all_metrics(preds_arr,labels_arr,thresh=0.0)
        gate_entropy_mean=float(np.mean(gate_entropy_list)) if gate_entropy_list else 0.0
        kl_align_mean=float(np.mean(kl_list)) if kl_list else 0.0
        pred_mean=float(preds_arr.mean()); pred_std=float(preds_arr.std())

        abs_err=np.abs(preds_arr-labels_arr)
        corr_err_logvar=0.0
        if len(tok_lv_means)==len(abs_err) and np.std(tok_lv_means)>1e-8 and abs_err.std()>1e-8:
            corr_err_logvar=float(np.corrcoef(abs_err,np.array(tok_lv_means))[0,1])

        w_batch_var_text=w_batch_var_audio=w_batch_var_vision=0.0
        w_text_mean=w_audio_mean=w_vision_mean=0.0
        w_text_std=w_audio_std=w_vision_std=0.0
        if weight_rows:
            w_all=np.concatenate(weight_rows,0)
            w_text_mean,w_audio_mean,w_vision_mean=[float(x) for x in w_all.mean(axis=0)]
            w_text_std,w_audio_std,w_vision_std=[float(x) for x in w_all.std(axis=0)]
            w_batch_var_text,w_batch_var_audio,w_batch_var_vision=[float(x) for x in w_all.var(axis=0)]
        elif self.prev_w_mean is not None:
            w_text_mean=float(self.prev_w_mean[0,0].item())
            w_audio_mean=float(self.prev_w_mean[0,1].item())
            w_vision_mean=float(self.prev_w_mean[0,2].item())

        stats={
            'loss':total_loss/max(1,step),
            'mae':metrics['mae'],'pcc':metrics['pcc'],'spr':metrics['spr'],'ccc':metrics['ccc'],
            'acc7':metrics['acc7'],'acc2_all':metrics['acc2_all'],'f1_all':metrics['f1_all'],
            'acc2_nz':metrics['acc2_nonzero'],'f1_nz':metrics['f1_nonzero'],
            'pred_mean':pred_mean,'pred_std':pred_std,
            'mae_base':float(np.mean(mae_base_list)),'mae_seq':float(np.mean(mae_seq_list)),
            'mae_feat':float(np.mean(mae_feat_list)),
            'parts_mae_std':float(np.std([np.mean(mae_base_list),
                                          np.mean(mae_seq_list),
                                          np.mean(mae_feat_list)])),
            'mean_token_logvar':float(np.mean(tok_lv_means)) if tok_lv_means else 0.0,
            'token_logvar_std':float(np.std(tok_lv_means)) if tok_lv_means else 0.0,
            'mean_token_precision':float(np.mean(tok_prec_means)) if tok_prec_means else 0.0,
            'token_precision_std':float(np.std(tok_prec_means)) if tok_prec_means else 0.0,
            'corr_err_logvar':corr_err_logvar,'gate_entropy':gate_entropy_mean,
            'entropy_band_pen':0.0,'entropy_clamp_pen':0.0,'kl_align':kl_align_mean,
            'w_text_mean':w_text_mean,'w_audio_mean':w_audio_mean,'w_vision_mean':w_vision_mean,
            'w_text_std':w_text_std,'w_audio_std':w_audio_std,'w_vision_std':w_vision_std,
            'w_batch_var_text':w_batch_var_text,'w_batch_var_audio':w_batch_var_audio,'w_batch_var_vision':w_batch_var_vision,
            'hard_gain':float(np.mean(hard_gain_list)),'easy_pen':float(np.mean(easy_pen_list)),
            'var_reg_loss':float(np.mean(var_reg_list)),'min_logvar_pull_loss':float(np.mean(min_pull_list)),
            'contrast_loss':float(np.mean(contrast_list)),'poscorr_loss':float(np.mean(poscorr_list)),
            'grad_head':self._last_gn[0],'grad_gating':self._last_gn[1],
            'grad_backbone':self._last_gn[2],'grad_av':self._last_gn[3],
            'preds':preds_arr if collect_preds else None,
            'labels':labels_arr if collect_preds else None,
            'sign_logits':np.concatenate(sign_logits,0) if (collect_preds and sign_logits) else None,
            'gating_mode':mode
        }
        return stats

    def _write_csv(self,epoch,split,s,thresh_used=0.0):
        def g(k,d=0.0): return float(s.get(k,d))
        with open(self.cfg.diag_csv,'a',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch,split,
                f"{g('mae'):.6f}",f"{g('pcc'):.6f}",f"{g('spr'):.6f}",f"{g('ccc'):.6f}",
                f"{g('acc7'):.6f}",f"{g('acc2_all'):.6f}",f"{g('f1_all'):.6f}",
                f"{g('acc2_nz'):.6f}",f"{g('f1_nz'):.6f}",
                f"{g('pred_mean'):.6f}",f"{g('pred_std'):.6f}",
                f"{g('mae_base'):.6f}",f"{g('mae_seq'):.6f}",f"{g('mae_feat'):.6f}",f"{g('parts_mae_std'):.6f}",
                f"{g('mean_token_logvar'):.6f}",f"{g('token_logvar_std'):.6f}",
                f"{g('mean_token_precision'):.6f}",f"{g('token_precision_std'):.6f}",
                f"{g('corr_err_logvar'):.6f}",f"{g('gate_entropy'):.6f}",
                f"{g('entropy_band_pen'):.6f}",f"{g('entropy_clamp_pen'):.6f}",f"{g('kl_align'):.6f}",
                f"{g('w_text_mean'):.6f}",f"{g('w_audio_mean'):.6f}",f"{g('w_vision_mean'):.6f}",
                f"{g('w_text_std'):.6f}",f"{g('w_audio_std'):.6f}",f"{g('w_vision_std'):.6f}",
                f"{g('w_batch_var_text'):.6f}",f"{g('w_batch_var_audio'):.6f}",f"{g('w_batch_var_vision'):.6f}",
                f"{g('hard_gain'):.6f}",f"{g('easy_pen'):.6f}",
                f"{g('var_reg_loss'):.6f}",f"{g('min_logvar_pull_loss'):.6f}",
                f"{g('contrast_loss'):.6f}",f"{g('poscorr_loss'):.6f}",
                f"{g('grad_head'):.6f}",f"{g('grad_gating'):.6f}",
                f"{g('grad_backbone'):.6f}",f"{g('grad_av'):.6f}",
                f"{thresh_used:.6f}",
                s.get('gating_mode','NA'),
                getattr(self.cfg,'experiment_tag','full')
            ])

    def _save_test_results_for_viz(self, test_s):
        """
        保存测试结果用于可视化（包含 Video ID, 原始文本, 真实标签, 预测值）
        生成 visualization_data.csv 方便人工筛选案例
        """
        import numpy as np
        import pandas as pd
        from transformers import BertTokenizer

        print("\n" + "=" * 70)
        print("[Viz] Collecting detailed test results for case study...")
        print("=" * 70)

        self.model.eval()

        # 初始化收集列表
        video_ids_list = []
        raw_texts = []

        # 加载 Tokenizer 用于还原文本
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except:
            # 如果没联网，尝试用本地路径或简单的 split 模拟（通常跑这个代码的环境都能连 HuggingFace）
            print("[Warning] Could not load BertTokenizer. Text might not be decoded.")
            tokenizer = None

        with torch.no_grad():
            for i, b in enumerate(self.test_loader):
                ids = b['input_ids'].to(self.device)

                # 1. 获取 Video ID
                if 'id' in b:
                    video_ids_list.extend(b['id'])
                else:
                    # 兜底：如果没有ID，生成索引ID
                    start_idx = len(video_ids_list)
                    video_ids_list.extend([f"sample_{start_idx + k}" for k in range(len(ids))])

                # 2. 解码文本
                if tokenizer:
                    for seq in ids:
                        # 将 ID 转回 Token
                        tokens = tokenizer.convert_ids_to_tokens(seq.cpu().numpy())
                        # 过滤特殊字符
                        clean_tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']]
                        # 简单的还原（BERT的分词会有 ## 前缀，这里简单处理一下方便阅读）
                        sentence = " ".join(clean_tokens).replace(" ##", "")
                        raw_texts.append(sentence)
                else:
                    raw_texts.extend(["[Tokenizer Error]"] * len(ids))

        # 3. 组装数据
        # test_s['preds'] 和 test_s['labels'] 是 numpy 数组
        # 确保长度一致
        n_samples = len(test_s['preds'])
        if len(video_ids_list) != n_samples:
            print(f"[Warning] ID count ({len(video_ids_list)}) != Preds count ({n_samples}). Truncating to minimum.")
            min_len = min(len(video_ids_list), n_samples)
            video_ids_list = video_ids_list[:min_len]
            raw_texts = raw_texts[:min_len]
            preds = test_s['preds'][:min_len]
            labels = test_s['labels'][:min_len]
        else:
            preds = test_s['preds']
            labels = test_s['labels']

        # 计算绝对误差，方便筛选
        abs_error = np.abs(preds - labels)

        # 创建 DataFrame
        df = pd.DataFrame({
            'Video_ID': video_ids_list,
            'Label': labels,
            'Prediction': preds,
            'Abs_Error': abs_error,
            'Text': raw_texts
        })

        # 4. 保存为 CSV
        csv_path = os.path.join(self.cfg.save_dir, 'visualization_data.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"[Viz] Visualization data saved to: {csv_path}")
        print(f"   - Total Samples: {len(df)}")
        print(f"   - CSV can be used to pick examples for the paper.")
        print("=" * 70 + "\n")

    def train(self):
        print(f"[RUN] tag={getattr(self.cfg,'experiment_tag','full')} mode={getattr(self.cfg,'gating_mode','dynamic')} save_dir={self.cfg.save_dir}")
        for epoch in range(1,int(getattr(self.cfg,'max_epochs',30))+1):
            self.current_epoch=epoch
            if epoch==getattr(self.cfg,'token_start_epoch',4):
                self._apply_warmdown_if_needed(epoch)
            self._schedule_weights(epoch)

            if self.use_ema and (not self.ema_active) and epoch>=getattr(self.cfg,'ema_start_epoch',5):
                self.ema_active=True
                if hasattr(self.ema,'_build_from_model'):
                    self.ema._build_from_model(self.model)
                self.ema_updates=0

            train_s=self._run_epoch(self.train_loader,epoch,True)
            torch.cuda.empty_cache()

            eval_state = None
            if self.use_ema and self.ema_active and self.ema_updates>=self.ema_min_updates_for_eval:
                self.ema.apply_to(self.model)
                val_s=self._run_epoch(self.valid_loader,epoch,False,collect_preds=True)
                if bool(getattr(self.cfg, 'save_eval_weights', True)):
                    eval_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.ema.restore(self.model)
            else:
                val_s=self._run_epoch(self.valid_loader,epoch,False,collect_preds=True)

            self._best_thresh=scan_best_thresh(val_s['preds'],val_s['labels'],
                                            lo=getattr(self.cfg,'thresh_scan_min',-0.35),
                                            hi=getattr(self.cfg,'thresh_scan_max',0.35),
                                            steps=getattr(self.cfg,'thresh_scan_steps',71))

            if bool(getattr(self.cfg,'cls_calib_enable',True)) and val_s.get('sign_logits') is not None:
                best_sign=_grid_search_sign(val_s['sign_logits'],val_s['labels'],
                                            acc_floor=getattr(self.cfg,'cls_calib_acc_floor',0.83),
                                            T_min=getattr(self.cfg,'cls_calib_T_min',0.6),
                                            T_max=getattr(self.cfg,'cls_calib_T_max',1.6),
                                            T_steps=getattr(self.cfg,'cls_calib_T_steps',21),
                                            thr_min=getattr(self.cfg,'cls_calib_thr_min',-0.4),
                                            thr_max=getattr(self.cfg,'cls_calib_thr_max',0.4),
                                            thr_steps=getattr(self.cfg,'cls_calib_thr_steps',81))
                self._sign_calib=best_sign
                print(f"[Calib] sign-head T={best_sign['T']:.2f} thr={best_sign['thr']:+.2f} Acc2(nz)={best_sign['acc']:.3f} F1(nz)={best_sign['f1']:.3f}")

            joint=self._joint_search(val_s)
            if joint:
                self._joint_calib=joint
                print(f"[JointCalib] T={joint['T']:.2f} thr_cls={joint['thr_cls']:+.2f} thr_reg={joint['thr_reg']:+.2f} "
                    f"macroF1={joint['macroF1']:.3f} F1_reg={joint['F1_nz']:.3f}")

            m0=compute_all_metrics(val_s['preds'],val_s['labels'],0.0)
            ms=compute_all_metrics(val_s['preds'],val_s['labels'],self._best_thresh)
            print(f"[Epoch {epoch:02d}] MAE{val_s['mae']:.3f} PCC{val_s['pcc']:.3f} F1(0){m0['f1_nonzero']:.3f} "
                f"F1(t*){ms['f1_nonzero']:.3f} GateH{val_s['gate_entropy']:.3f} "
                f"wT/A/V {val_s['w_text_mean']:.2f}/{val_s['w_audio_mean']:.2f}/{val_s['w_vision_mean']:.2f} "
                f"hardGain{val_s['hard_gain']:.3f} easyPen{val_s['easy_pen']:.3f} mode={val_s['gating_mode']}")

            self._write_csv(epoch,'train',train_s,thresh_used=0.0)
            self._write_csv(epoch,'valid',val_s,thresh_used=self._best_thresh)

            criterion = getattr(self.cfg, 'select_best_by', 'composite')
            if not hasattr(self, '_best_key'):
                self._best_key = float('inf') if criterion in ['composite','mae'] else -float('inf')

            if criterion == 'mae':
                key = val_s['mae']
                improved = (key < self._best_key - 1e-4)
            elif criterion == 'f1':
                key = val_s['f1_nz']
                improved = (key > self._best_key + 1e-4)
            elif criterion == 'pcc':
                key = val_s['pcc']
                improved = (key > self._best_key + 1e-4)
            elif criterion == 'ccc':
                key = val_s['ccc']
                improved = (key > self._best_key + 1e-4)
            else:
                key = self._composite_val(val_s)
                improved = (key < self._best_key - 1e-4)

            if improved:
                self._best_key = key
                torch.save({
                    'model': eval_state if eval_state is not None else self.model.state_dict(),
                    'epoch': epoch,
                    'val_mae': val_s['mae'],
                    'val_pcc': val_s['pcc'],
                    'val_ccc': val_s['ccc'],
                    't_star': self._best_thresh,
                    'r_vec_ema': self.r_vec_ema.detach().cpu() if hasattr(self,'r_vec_ema') and self.r_vec_ema is not None else None,
                    'joint': self._joint_calib
                }, os.path.join(self.cfg.save_dir, 'best.pth'))
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= int(getattr(self.cfg, 'patience', 8)):
                    print("Early stopping."); break

        # Test the checkpoint selected on validation, not just the final epoch.
        best_path = os.path.join(self.cfg.save_dir, 'best.pth')
        loaded_best = False
        if bool(getattr(self.cfg, 'load_best_for_test', True)) and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            self._best_thresh = float(ckpt.get('t_star', self._best_thresh))
            self._joint_calib = ckpt.get('joint', self._joint_calib)
            if ckpt.get('r_vec_ema') is not None:
                self.r_vec_ema = ckpt['r_vec_ema'].to(self.device)
            loaded_best = True
            print(f"[LoadBest] epoch={ckpt.get('epoch')} val_mae={ckpt.get('val_mae', float('nan')):.4f}")

        if (not loaded_best) and self.use_ema and self.ema_active and self.ema_updates>=self.ema_min_updates_for_eval:
            self.ema.apply_to(self.model)
            test_s=self._run_epoch(self.test_loader,self.cfg.max_epochs,False,collect_preds=True)
            self.ema.restore(self.model)
        else:
            test_s=self._run_epoch(self.test_loader,self.cfg.max_epochs,False,collect_preds=True)
        
        # 🔥 保存可视化数据
        self._save_test_results_for_viz(test_s)

        m0=compute_all_metrics(test_s['preds'],test_s['labels'],0.0)
        ms=compute_all_metrics(test_s['preds'],test_s['labels'],self._best_thresh)
        print(f"[TEST] t=0 | MAE:{m0['mae']:.3f} F1:{m0['f1_nonzero']:.3f} Acc2:{m0['acc2_nonzero']:.3f}")
        print(f"[TEST] t*={self._best_thresh:+.3f} | MAE:{ms['mae']:.3f} F1:{ms['f1_nonzero']:.3f} Acc2:{ms['acc2_nonzero']:.3f}")

        if self._sign_calib and test_s.get('sign_logits') is not None:
            T=self._sign_calib['T']; thr=self._sign_calib['thr']
            acc_cls,f1_cls=_cls_metrics_from_logits(test_s['sign_logits'],test_s['labels'],T=T,thr=thr,nonzero=True)
            print(f"[TEST] sign-head tuned | Acc2(nz):{acc_cls:.3f} F1(nz):{f1_cls:.3f} (T={T:.2f} thr={thr:+.2f})")

        if self._joint_calib and test_s.get('sign_logits') is not None:
            jc=self._joint_calib
            logits=test_s['sign_logits']/jc['T']
            cls_pred=(logits>jc['thr_cls'])
            y=test_s['labels']; nz=(y!=0); gt=(y[nz]>0); pr=cls_pred[nz]
            tp=np.sum(pr & gt); fp=np.sum(pr & (~gt)); fn=np.sum((~pr)&gt); tn=np.sum((~pr)&(~gt))
            prec_pos=tp/(tp+fp+1e-8); rec_pos=tp/(tp+fn+1e-8)
            f1_pos=2*prec_pos*rec_pos/(prec_pos+rec_pos+1e-8)
            prec_neg=tn/(tn+fn+1e-8); rec_neg=tn/(tn+fp+1e-8)
            f1_neg=2*prec_neg*rec_neg/(prec_neg+rec_neg+1e-8)
            macroF1=0.5*(f1_pos+f1_neg)
            m_joint=compute_all_metrics(test_s['preds'],test_s['labels'],jc['thr_reg'])
            print(f"[TEST] joint | macroF1_cls:{macroF1:.3f} F1_reg:{m_joint['f1_nonzero']:.3f} Acc2_reg:{m_joint['acc2_nonzero']:.3f}")

        self._write_csv(-1,'test',test_s,thresh_used=self._best_thresh)
