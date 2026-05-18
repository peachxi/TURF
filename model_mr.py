import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import contextlib

class EMAWeights:
    def __init__(self, model, decay=0.997):
        self.decay=float(decay)
        self.shadow={}
        self.backup={}
        self._build_from_model(model)

    @torch.no_grad()
    def _build_from_model(self, model):
        self.shadow.clear()
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n]=p.detach().clone()

    @torch.no_grad()
    def _sync_new_params(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            need_add=(n not in self.shadow) or (self.shadow[n].shape!=p.shape) or \
                     (self.shadow[n].device!=p.device) or (self.shadow[n].dtype!=p.dtype)
            if need_add:
                self.shadow[n]=p.detach().clone()
        current={n for n,_ in model.named_parameters()}
        for k in list(self.shadow.keys()):
            if k not in current:
                del self.shadow[k]

    @torch.no_grad()
    def update(self, model):
        self._sync_new_params(model)
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0-self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self._sync_new_params(model)
        self.backup={}
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[n]=p.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup={}

class GatingNet(nn.Module):
    def __init__(self,in_dim,hidden,num_modal=3):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden)
        self.ln1=nn.LayerNorm(hidden)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden,num_modal)
    def forward(self,x,temp:float=1.0):
        logits=self.fc2(self.act(self.ln1(self.fc1(x))))
        T=max(1e-6,float(temp))
        return F.softmax(logits/T,dim=-1)

class LearnableTemporalPooling(nn.Module):
    def __init__(self,hidden,dropout=0.1):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden,max(8,hidden//2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8,hidden//2),1)
        )
    def forward(self,x):
        score=self.mlp(x).squeeze(-1)
        alpha=F.softmax(score,dim=1)
        pooled=torch.bmm(alpha.unsqueeze(1),x).squeeze(1)
        return pooled,alpha

def layer_pool_last_n(hidden_states,n=4):
    if hidden_states is None or len(hidden_states)<n:
        return hidden_states[-1]
    return torch.stack(hidden_states[-n:],dim=0).mean(dim=0)

def temporal_downsample(x,stride):
    if stride<=1: return x
    B,T,C=x.shape
    pad=(stride - (T%stride))%stride
    if pad>0:
        x=F.pad(x,(0,0,0,pad))
        T=x.size(1)
    x=x.view(B,T//stride,stride,C).mean(dim=2)
    return x

class MRModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        H=cfg.hidden_dim

        # Flags
        self.use_audio=bool(getattr(cfg,'use_audio',True))
        self.use_vision=bool(getattr(cfg,'use_vision',True))
        self.hard_drop_audio=bool(getattr(cfg,'hard_drop_audio',False))
        self.hard_drop_vision=bool(getattr(cfg,'hard_drop_vision',False))
        self.text_dropout_p=float(getattr(cfg,'text_dropout_p',0.0))
        self.text_dropout_until=int(getattr(cfg,'text_dropout_until_epoch',0))

        self.bert=AutoModel.from_pretrained(cfg.bert_model, output_hidden_states=True)
        if hasattr(self.bert,'config') and hasattr(self.bert.config,'use_cache'):
            self.bert.config.use_cache=False

        if cfg.freeze_text:
            self.bert.eval()
            for p in self.bert.parameters(): p.requires_grad=False
        else:
            if hasattr(self.bert,"gradient_checkpointing_enable"):
                self.bert.gradient_checkpointing_enable()
                if bool(getattr(cfg, 'strict_partial_unfreeze', False)) and hasattr(self.bert, "enable_input_require_grads"):
                    self.bert.enable_input_require_grads()
            if bool(getattr(cfg, 'strict_partial_unfreeze', False)):
                for p in self.bert.parameters():
                    p.requires_grad = False
            enc=getattr(self.bert,'encoder',getattr(self.bert,'transformer',None))
            if enc is not None and hasattr(enc,'layer'):
                total=len(enc.layer)
                freeze_until=max(0,total - cfg.unfreeze_last_n)
                for i in range(freeze_until):
                    for p in enc.layer[i].parameters():
                        p.requires_grad=False
                if bool(getattr(cfg, 'strict_partial_unfreeze', False)):
                    for i in range(freeze_until, total):
                        for p in enc.layer[i].parameters():
                            p.requires_grad = True

        self.text_proj=nn.Linear(self.bert.config.hidden_size,H)
        self.text_ln=nn.LayerNorm(H)

        # Token uncertainty
        self.use_token_uncertainty=cfg.use_token_uncertainty
        if self.use_token_uncertainty and cfg.use_token_bilstm:
            th=cfg.token_lstm_hidden
            self.token_lstm=nn.LSTM(H,th,batch_first=True,bidirectional=True)
            self.token_lstm_proj=nn.Linear(2*th,H)
        self.token_head=nn.Sequential(
            nn.Linear(H,H), nn.GELU(),
            nn.Dropout(cfg.token_head_dropout),
            nn.Linear(H,2)
        )

        # AV projection (lazy actual dims)
        self.av_hidden_dim=cfg.av_hidden_dim
        self.audio_in=None
        self.vision_in=None

        self.use_av_bilstm=cfg.use_av_bilstm
        if self.use_av_bilstm:
            self.a_lstm=nn.LSTM(self.av_hidden_dim,cfg.av_lstm_hidden,batch_first=True,bidirectional=False)
            self.v_lstm=nn.LSTM(self.av_hidden_dim,cfg.av_lstm_hidden,batch_first=True,bidirectional=False)
            self.a_proj=nn.Linear(cfg.av_lstm_hidden,H)
            self.v_proj=nn.Linear(cfg.av_lstm_hidden,H)
        else:
            self.a_proj=nn.Linear(self.av_hidden_dim,H)
            self.v_proj=nn.Linear(self.av_hidden_dim,H)

        self.use_ltp_audio=cfg.use_ltp_audio
        self.use_ltp_vision=cfg.use_ltp_vision
        if self.use_ltp_audio:  self.ltp_a=LearnableTemporalPooling(H,dropout=cfg.ltp_dropout)
        if self.use_ltp_vision: self.ltp_v=LearnableTemporalPooling(H,dropout=cfg.ltp_dropout)

        # Optional LayerNorm for modality pooled features (helps gating stability)
        self.a_ln=nn.LayerNorm(H)
        self.v_ln=nn.LayerNorm(H)

        # Sentence-level heads per modality
        def head():
            return nn.Sequential(nn.Linear(H,H), nn.ReLU(), nn.Linear(H,2))
        self.head_t=head(); self.head_a=head(); self.head_v=head()

        # Fusion MLP and sign head
        self.y_feat_mlp=nn.Sequential(
            nn.Linear(3*H + 1,H), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(H,1)
        )
        self.sign_head=nn.Sequential(
            nn.Linear(3*H + 1,H//2), nn.GELU(),
            nn.Linear(H//2,1)
        )

        # Dynamic gating net (unused in hard modes but kept for completeness)
        self.gating=GatingNet(3*H,int(1.0*H),3)

        # Static learnable logits (only used if not strict hard constant)
        if getattr(cfg,'gating_mode','dynamic')=='static_learned_hard':
            # Register as buffer to enforce strict constant (freeze)
            self.register_buffer('static_gate_weights', torch.tensor([1/3,1/3,1/3],dtype=torch.float))
        else:
            self.static_gate_logits=nn.Parameter(torch.zeros(3))

        self.logvar_min=cfg.logvar_min; self.logvar_max=cfg.logvar_max
        self.token_logvar_min = getattr(cfg, 'token_logvar_min', self.logvar_min)
        self.token_logvar_max = getattr(cfg, 'token_logvar_max', self.logvar_max)
        self.text_logvar_min = getattr(cfg, 'text_logvar_min', self.logvar_min)
        self.text_logvar_max = getattr(cfg, 'text_logvar_max', self.logvar_max)
        self.modality_logvar_min = getattr(cfg, 'modality_logvar_min', self.logvar_min)
        self.modality_logvar_max = getattr(cfg, 'modality_logvar_max', self.logvar_max)

    def _bert_forward(self,input_ids,attention_mask):
        ctx=torch.no_grad() if self.cfg.freeze_text else contextlib.nullcontext()
        with ctx:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    def _ensure_modal_projs(self,a,v):
        Da=int(a.shape[-1]); Dv=int(v.shape[-1])
        if self.audio_in is None or self.audio_in.in_features!=Da:
            self.audio_in=nn.Linear(Da,self.av_hidden_dim,bias=True).to(a.device)
        if self.vision_in is None or self.vision_in.in_features!=Dv:
            self.vision_in=nn.Linear(Dv,self.av_hidden_dim,bias=True).to(v.device)

    def _maybe_text_dropout(self,t_feat,mu_text_seq,epoch):
        if self.training and self.text_dropout_p>0.0 and epoch is not None and epoch<=self.text_dropout_until:
            B=t_feat.size(0)
            m=torch.bernoulli(torch.full((B,1),1.0-self.text_dropout_p,device=t_feat.device))
            t_feat=t_feat*m; mu_text_seq=mu_text_seq*m
        return t_feat,mu_text_seq

    def forward(self,input_ids,attention_mask,token_type_ids,audio,vision,epoch=None):
        self._ensure_modal_projs(audio,vision)
        bert_out=self._bert_forward(input_ids,attention_mask)
        txt_all=layer_pool_last_n(bert_out.hidden_states,4) if self.cfg.use_layer_pool else bert_out.last_hidden_state
        tok=self.text_ln(self.text_proj(txt_all))

        if self.use_token_uncertainty and hasattr(self,'token_lstm'):
            ctx,_=self.token_lstm(tok)
            tok_ctx=self.token_lstm_proj(ctx)+tok
        else:
            tok_ctx=tok

        token_out=self.token_head(tok_ctx)
        mu_tokens=token_out[...,0]
        lv_raw=token_out[...,1]
        lv_tokens=self.token_logvar_min + (self.token_logvar_max - self.token_logvar_min)*torch.sigmoid(lv_raw) if self.cfg.token_logvar_bound else lv_raw
        mask=attention_mask.float()
        precision_tokens=torch.exp(-lv_tokens)
        if self.cfg.precision_temp!=1.0:
            precision_tokens=precision_tokens.pow(self.cfg.precision_temp)
        prec_for_agg=precision_tokens.detach() if self.cfg.precision_detach else precision_tokens
        denom=(prec_for_agg*mask).sum(dim=1,keepdim=True).clamp_min(1e-8)
        mu_text_seq=((prec_for_agg*mask)*mu_tokens).sum(dim=1,keepdim=True)/denom
        r_text=(precision_tokens*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1.0)

        t_feat=tok_ctx[:,0,:]
        t_feat,mu_text_seq=self._maybe_text_dropout(t_feat,mu_text_seq,epoch)
        out_t=self.head_t(t_feat); mu_t=out_t[:,0:1]; lv_t=out_t[:,1:2].clamp(self.text_logvar_min,self.text_logvar_max)

        a=self.audio_in(audio); v=self.vision_in(vision)
        a=temporal_downsample(a,self.cfg.av_downsample_stride)
        v=temporal_downsample(v,self.cfg.av_downsample_stride)
        if self.use_av_bilstm:
            a,_=self.a_lstm(a); v,_=self.v_lstm(v)
            a=self.a_proj(a); v=self.v_proj(v)
        else:
            a=self.a_proj(a); v=self.v_proj(v)
        a_pool=self.ltp_a(a)[0] if self.use_ltp_audio else a.mean(dim=1)
        v_pool=self.ltp_v(v)[0] if self.use_ltp_vision else v.mean(dim=1)

        # LayerNorm pooled features
        a_pool=self.a_ln(a_pool)
        v_pool=self.v_ln(v_pool)

        out_a=self.head_a(a_pool); mu_a=out_a[:,0:1]; lv_a=out_a[:,1:2].clamp(self.modality_logvar_min,self.modality_logvar_max)
        out_v=self.head_v(v_pool); mu_v=out_v[:,0:1]; lv_v=out_v[:,1:2].clamp(self.modality_logvar_min,self.modality_logvar_max)

        if not (self.use_audio and not self.hard_drop_audio):
            a_pool=torch.zeros_like(a_pool); mu_a=torch.zeros_like(mu_a); lv_a=torch.full_like(lv_a,self.modality_logvar_max)
        if not (self.use_vision and not self.hard_drop_vision):
            v_pool=torch.zeros_like(v_pool); mu_v=torch.zeros_like(mu_v); lv_v=torch.full_like(lv_v,self.modality_logvar_max)

        mus=torch.cat([mu_t,mu_a,mu_v],dim=1)
        logvars=torch.cat([lv_t,lv_a,lv_v],dim=1)

        feats_cat=torch.cat([t_feat,a_pool,v_pool],dim=1)
        w_content=self.gating(feats_cat,temp=self.cfg.gate_temp)  # 未必使用

        precision_a=torch.exp(-lv_a.squeeze(1))
        precision_v=torch.exp(-lv_v.squeeze(1))
        r_vec=torch.stack([r_text,precision_a,precision_v],dim=1)

        y_feat=self.y_feat_mlp(torch.cat([feats_cat,mu_text_seq],dim=1)).squeeze(1)
        sign_logit=self.sign_head(torch.cat([feats_cat,mu_text_seq],dim=1)).squeeze(1)

        avg_token_logvar=((lv_tokens*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1.0)).unsqueeze(1)
        u_text=0.5*(avg_token_logvar+lv_t)
        u_modal=torch.cat([u_text,lv_a,lv_v],dim=1)

        gmask=feats_cat.new_tensor([
            1.0,
            1.0 if (self.use_audio and not self.hard_drop_audio) else 0.0,
            1.0 if (self.use_vision and not self.hard_drop_vision) else 0.0
        ]).unsqueeze(0).expand(feats_cat.size(0),-1)

        mode=getattr(self.cfg,'gating_mode','dynamic')

        if mode=='static_learned_hard':
            w_static=self.static_gate_weights.unsqueeze(0).expand(feats_cat.size(0),-1)
        else:
            w_static=torch.softmax(self.static_gate_logits,dim=0).unsqueeze(0).expand(feats_cat.size(0),-1)

        return {
            'mus':mus,'logvars':logvars,
            'weights_content':w_content,'weights_static':w_static,
            'gate_mask':gmask,
            'mu_tokens':mu_tokens,'logvar_tokens':lv_tokens,'mu_text_seq':mu_text_seq,
            'r_vec':r_vec,'mask':mask,
            'y_feat':y_feat,'sign_logit':sign_logit,
            'u_modal':u_modal,
            'fusion_features': feats_cat
        }
