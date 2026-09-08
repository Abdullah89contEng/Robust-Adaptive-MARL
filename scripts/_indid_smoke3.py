import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from method.io import load_phase1
from method.training.phase2 import Phase2Trainer, Phase2Config

tr = load_phase1("runs/retrain_v3_phys/phase1_checkpoint_14999.pt")
cfg = Phase2Config()
print(f"rollouts_per_step={cfg.rollouts_per_step} len_segment={cfg.len_segment} lr={cfg.lr}")
t2 = Phase2Trainer(tr, cfg)
for it in range(50):
    s = t2.train_iteration()
    if it % 10 == 0 or it == 49:
        pres, posts = [], []
        with torch.no_grad():
            for _ in range(8):
                fs, stt = t2.collect_labeled_episode()
                pr = t2.detector(fs[:, 0]); sw = int(stt[0])
                if sw < pr.shape[1] - 3:
                    pres.append(float(pr[:, max(sw-8,0):sw].mean()))
                    posts.append(float(pr[:, sw:sw+15].mean()))
        pre, post = (np.mean(pres) if pres else float("nan")), (np.mean(posts) if posts else float("nan"))
        print(f"  it{it:3d}  l_cpd={s['l_cpd']:8.2f} grad={s['grad_norm']:6.1f}  p_pre={pre:.3f} p_post={post:.3f} gap={post-pre:+.3f}")
