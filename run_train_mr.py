import torch
from config_mr import ConfigMR
from model_mr import MRModel
from trainer_mr import HybridTrainer
from dataloader_processed import get_dataloaders

def infer_modal_dims_from_loader(loader):
    for b in loader:
        a = b['audio']      # [B, Ta, Da]
        v = b['vision']     # [B, Tv, Dv]
        return int(a.shape[-1]), int(v.shape[-1])
    return None, None

def main():
    cfg = ConfigMR()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    print(f"[CUDA] available={torch.cuda.is_available()} device={cfg.device}")

    train_loader, valid_loader, test_loader = get_dataloaders(cfg)

    a_dim, v_dim = infer_modal_dims_from_loader(train_loader)
    if a_dim is not None and getattr(cfg, 'audio_dim', None) != a_dim:
        print(f"[Info] Overriding cfg.audio_dim: {getattr(cfg, 'audio_dim', None)} -> {a_dim}")
        cfg.audio_dim = a_dim
    if v_dim is not None and getattr(cfg, 'vision_dim', None) != v_dim:
        print(f"[Info] Overriding cfg.vision_dim: {getattr(cfg, 'vision_dim', None)} -> {v_dim}")
        cfg.vision_dim = v_dim

    model = MRModel(cfg)

    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total:{tot/1e6:.2f}M trainable:{trn/1e6:.2f}M")

    trainer = HybridTrainer(model, (train_loader, valid_loader, test_loader), cfg)
    trainer.train()

if __name__ == '__main__':
    main()