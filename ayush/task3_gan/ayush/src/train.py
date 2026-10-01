"""Train unpaired CycleGAN. Smoke flag is the gate before the 40-epoch run."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.runtime import load_config, open_run_log, peak_mem_gb, resolve_device, set_seeds, write_manifest
from task3_gan.ayush.src.datasets import ImageFolderList, load_paths, write_photo_lists
from task3_gan.ayush.src.networks import ImagePool, PatchDiscriminator, ResnetGenerator, init_weights


def lr_factor(epoch: int, n_const: int = 25, n_decay: int = 15) -> float:
    """Full rate for n_const epochs, then linear decay. The last epoch is still above 0."""
    return max(0.0, 1.0 - max(0, epoch + 1 - n_const) / float(n_decay + 1))


def diffaug(x: torch.Tensor) -> torch.Tensor:
    """Translation and cutout on discriminator inputs."""
    b, _, h, w = x.shape
    sy, sx = max(1, int(h * 0.125)), max(1, int(w * 0.125))
    dx = torch.randint(-sx, sx + 1, (b,), device=x.device)
    dy = torch.randint(-sy, sy + 1, (b,), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=(int(dy[i]), int(dx[i])), dims=(1, 2)) for i in range(b)])
    cut = torch.ones_like(x)
    ch, cw = max(1, h // 2), max(1, w // 2)
    for i in range(b):
        y0 = int(torch.randint(0, h, ()))
        x0 = int(torch.randint(0, w, ()))
        cut[i, :, max(0, y0 - ch // 2) : min(h, y0 + ch // 2), max(0, x0 - cw // 2) : min(w, x0 + cw // 2)] = 0
    return x * cut


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    for shadow, src in zip(ema.parameters(), model.parameters()):
        shadow.mul_(decay).add_(src, alpha=1.0 - decay)


def choose_batch(total_gb: float, requested: int) -> int:
    """Fill the GPU. A 32 GB RTX 5090 takes 8; pass --batch-size to override."""
    if requested > 0:
        return requested
    if total_gb >= 70:
        return 16
    if total_gb >= 28:
        return 8
    if total_gb >= 14:
        return 4
    return 1


def gan_loss(pred: torch.Tensor, real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def save_grid(path: Path, images: list[torch.Tensor]) -> None:
    tiles = []
    for im in images:
        x = (im[0].detach().cpu().clamp(-1, 1) + 1) * 0.5
        tiles.append((x.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    canvas = Image.new("RGB", (256 * len(tiles), 256))
    for i, tile in enumerate(tiles):
        canvas.paste(Image.fromarray(tile), (256 * i, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ayush/base.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all-photos", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--decay-start", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    args = ap.parse_args()
    cfg, cfg_hash = load_config(args.config)
    set_seeds(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type != "cuda":
        raise SystemExit(f"refusing to train task 3 on {device}")
    log = open_run_log(cfg, "task3", "cyclegan", cfg_hash)
    t3 = cfg.task3
    repo = REPO
    member = repo / "task3_gan" / cfg.member
    ckpt_dir = member / "checkpoints"
    out_dir = member / "outputs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 2 if args.smoke else (args.epochs or int(t3.epochs))
    n_const = 2 if args.smoke else (args.decay_start or int(t3.decay_start_epoch))
    n_decay = max(1, epochs - n_const)
    limit = 20 if args.smoke else None
    data_dir = repo / cfg.paths.task3_data
    if args.all_photos and not args.smoke and not (data_dir / "photo_train.txt").exists():
        write_photo_lists(data_dir, int(cfg.seed))
    monets, photos = load_paths(
        data_dir, data_dir / "photo_subset_2000.txt", limit, all_photos=args.all_photos and not args.smoke,
    )
    log.info(
        "monets=%s photos=%s epochs=%s smoke=%s all_photos=%s resume=%s",
        len(monets), len(photos), epochs, int(args.smoke), int(args.all_photos and not args.smoke), args.resume or "none",
    )
    gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    batch = 1 if args.smoke else choose_batch(gb, args.batch_size)
    workers = 0 if args.smoke else min(12, max(2, (os.cpu_count() or 4) - 2))
    loader_kw = dict(
        batch_size=batch, shuffle=True, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
        **({"prefetch_factor": 4} if workers > 0 else {}),
    )
    monet_loader = DataLoader(ImageFolderList(monets, train=True), **loader_kw)
    photo_loader = DataLoader(ImageFolderList(photos, train=True), **loader_kw)
    log.info("device=%s mem_gb=%.1f batch=%s workers=%s amp=bf16", torch.cuda.get_device_name(device), gb, batch, workers)

    g_ab = ResnetGenerator().to(device)
    g_ba = ResnetGenerator().to(device)
    d_a = PatchDiscriminator().to(device)
    d_b = PatchDiscriminator().to(device)
    for net in (g_ab, g_ba, d_a, d_b):
        init_weights(net)
    carried = 0
    if args.resume and not args.smoke:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        g_ab.load_state_dict(ckpt["g_ab"])
        g_ba.load_state_dict(ckpt["g_ba"])
        d_a.load_state_dict(ckpt["d_a"])
        d_b.load_state_dict(ckpt["d_b"])
        carried = int(ckpt["epoch"])
        log.info("resumed epoch=%s from %s", carried, args.resume)
    ema_ab = ResnetGenerator().to(device).eval()
    ema_ba = ResnetGenerator().to(device).eval()
    ema_src_ab = ckpt["ema_g_ab"] if args.resume and not args.smoke and "ema_g_ab" in ckpt else g_ab.state_dict()
    ema_src_ba = ckpt["ema_g_ba"] if args.resume and not args.smoke and "ema_g_ba" in ckpt else g_ba.state_dict()
    ema_ab.load_state_dict(ema_src_ab)
    ema_ba.load_state_dict(ema_src_ba)
    for p in itertools.chain(ema_ab.parameters(), ema_ba.parameters()):
        p.requires_grad_(False)
    n_params = sum(p.numel() for net in (g_ab, g_ba, d_a, d_b) for p in net.parameters())
    log.info("params_total=%s", n_params)

    opt_g = torch.optim.Adam(itertools.chain(g_ab.parameters(), g_ba.parameters()), lr=float(t3.lr), betas=(float(t3.beta1), 0.999))
    opt_d = torch.optim.Adam(itertools.chain(d_a.parameters(), d_b.parameters()), lr=float(t3.lr), betas=(float(t3.beta1), 0.999))
    pool_a, pool_b = ImagePool(int(t3.pool_size)), ImagePool(int(t3.pool_size))
    history = []
    nan_count = 0
    gmax = dmax = 0.0
    t0 = time.time()
    images_seen = 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    for net in (g_ab, g_ba, d_a, d_b, ema_ab, ema_ba):
        net.to(memory_format=torch.channels_last)

    for epoch in range(epochs):
        factor = lr_factor(epoch, n_const, n_decay)
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = float(t3.lr) * factor
        sums = {k: torch.zeros((), device=device) for k in ("g", "d_a", "d_b", "cycle", "identity")}
        steps = 0
        monet_iter = iter(monet_loader)

        def take_monet(it):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(monet_loader)
                batch = next(it)
            return batch, it

        for photo, _stem in photo_loader:
            (monet, _), monet_iter = take_monet(monet_iter)
            real_a = monet.to(device, non_blocking=True, memory_format=torch.channels_last)
            real_b = photo.to(device, non_blocking=True, memory_format=torch.channels_last)
            b = real_a.shape[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                fake_b = g_ab(real_a)
                fake_a = g_ba(real_b)
                loss_gan = gan_loss(d_b(diffaug(fake_b)), True) + gan_loss(d_a(diffaug(fake_a)), True)
                loss_cycle = F.l1_loss(g_ba(fake_b), real_a) + F.l1_loss(g_ab(fake_a), real_b)
                loss_idt = F.l1_loss(g_ab(real_b), real_b) + F.l1_loss(g_ba(real_a), real_a)
                loss_g = loss_gan + float(t3.lambda_cycle) * loss_cycle + (float(t3.lambda_identity) * float(t3.lambda_cycle)) * loss_idt
            if not torch.isfinite(loss_g):
                nan_count += 1
                log.info("nan loss_G step=%s total=%s", steps, nan_count)
                if nan_count > 10:
                    raise SystemExit("too many NaNs")
                continue
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(itertools.chain(g_ab.parameters(), g_ba.parameters()), 100.0)
            opt_g.step()
            ema_update(ema_ab, g_ab)
            ema_update(ema_ba, g_ba)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_d_a = 0.5 * (gan_loss(d_a(diffaug(real_a)), True) + gan_loss(d_a(diffaug(pool_a.query(fake_a.detach()))), False))
                loss_d_b = 0.5 * (gan_loss(d_b(diffaug(real_b)), True) + gan_loss(d_b(diffaug(pool_b.query(fake_b.detach()))), False))
            opt_d.zero_grad(set_to_none=True)
            (loss_d_a + loss_d_b).backward()
            dnorm = torch.nn.utils.clip_grad_norm_(itertools.chain(d_a.parameters(), d_b.parameters()), 100.0)
            opt_d.step()

            sums["g"] = sums["g"] + loss_g.detach()
            sums["d_a"] = sums["d_a"] + loss_d_a.detach()
            sums["d_b"] = sums["d_b"] + loss_d_b.detach()
            sums["cycle"] = sums["cycle"] + loss_cycle.detach()
            sums["identity"] = sums["identity"] + loss_idt.detach()
            steps += 1
            images_seen += b * 2
            gmax = max(gmax, float(gnorm))
            dmax = max(dmax, float(dnorm))
            if steps % 100 == 0 or steps == 1:
                log.info(
                    "epoch=%s step=%s loss_G=%.4f loss_D_A=%.4f loss_D_B=%.4f cycle=%.4f idt=%.4f gnorm=%.3f dnorm=%.3f nan=%s",
                    epoch + 1, steps, float(loss_g), float(loss_d_a), float(loss_d_b), float(loss_cycle), float(loss_idt), float(gnorm), float(dnorm), nan_count,
                )
        row = {k: float(v) / max(1, steps) for k, v in sums.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        log.info("epoch_mean=%s", row)
        g_ab.eval()
        with torch.no_grad():
            ra, _ = next(iter(monet_loader))
            rb, _ = next(iter(photo_loader))
            ra, rb = ra.to(device), rb.to(device)
            save_grid(out_dir / "samples" / f"epoch_{epoch+1:03d}.png", [ra, ema_ab(ra), rb, ema_ba(rb)])
        g_ab.train()
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            seen = carried + epoch + 1
            torch.save({
                "g_ab": g_ab.state_dict(), "g_ba": g_ba.state_dict(),
                "ema_g_ab": ema_ab.state_dict(), "ema_g_ba": ema_ba.state_dict(),
                "d_a": d_a.state_dict(), "d_b": d_b.state_dict(),
                "epoch": seen, "history": history,
            }, ckpt_dir / f"cyclegan_epoch{seen}.pt")

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("g", "d_a", "cycle", "identity"):
        ax.plot([h["epoch"] for h in history], [h[key] for h in history], label=key)
    ax.legend()
    ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)
    seconds = time.time() - t0
    metrics = {
        "final_cycle_loss": history[-1]["cycle"],
        "final_identity_loss": history[-1]["identity"],
        "grad_norm_g_max": gmax,
        "grad_norm_d_max": dmax,
        "nan_count": nan_count,
        "param_count_total": n_params,
        "train_seconds": seconds,
        "images_per_sec": images_seen / max(1e-6, seconds),
        "peak_mem_gb": peak_mem_gb(),
        "epochs": carried + epochs,
        "smoke": int(args.smoke),
        "photos": len(photos),
        "resumed_from": carried,
        "batch_size": batch,
    }
    path = member / ("train_metrics_full.csv" if args.all_photos else "train_metrics.csv")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    write_manifest(
        cfg, 3, cfg_hash,
        checkpoint=str((ckpt_dir / f"cyclegan_epoch{carried + epochs}.pt").relative_to(repo)),
        duration_sec=seconds,
        metric_rows={"train_metrics.csv": 1},
        extra={"smoke": int(args.smoke)},
    )
    log.info("done metrics=%s", metrics)


if __name__ == "__main__":
    main()
