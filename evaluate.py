import os
import glob
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader

from sasnet import StructureEncoder
from stage2_fusion import CrossAttentionFusion
from decoder import SegmentationDecoder
from slum_dataloader import BanglaSlumDataset


# ==========================================
# CONFIG
# ==========================================
DATA_DIR    = os.path.join(os.getcwd(), "dhaka_dataset")
CKPT_DIR    = os.path.join(os.getcwd(), "checkpoints_stage2")
RESULTS_DIR = os.path.join(os.getcwd(), "eval_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FUSION_CKPT  = os.path.join(CKPT_DIR, "best_fusion.pth")
DECODER_CKPT = os.path.join(CKPT_DIR, "best_decoder.pth")
STRUCT_CKPT  = os.path.join(os.getcwd(), "checkpoints_stage1", "best_sasnet.pth")

THRESHOLD    = 0.5
BATCH_SIZE   = 1     # 1 tile at a time for clean per-tile reporting

# Tile IDs that are formal-dense control regions (paper Section 4.2.1)
# These are the tiles where we specifically measure false-positive rate
FORMAL_CONTROL_TILES = {
    "old_dhaka",
    "gulshan",
    "baridhara",
}


# ==========================================
# MODEL LOADING
# ==========================================
def load_models(device):
    struct_enc = StructureEncoder().to(device)
    fusion     = CrossAttentionFusion(num_se_channels=3).to(device)
    decoder    = SegmentationDecoder().to(device)

    for ckpt, model, name in [
        (STRUCT_CKPT,  struct_enc, "StructureEncoder"),
        (FUSION_CKPT,  fusion,     "Fusion"),
        (DECODER_CKPT, decoder,    "Decoder"),
    ]:
        if ckpt and os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"  ✅ {name}: {os.path.basename(ckpt)}")
        else:
            print(f"  ⚠️  {name}: no checkpoint — using random weights")

    struct_enc.eval()
    fusion.eval()
    decoder.eval()
    return struct_enc, fusion, decoder


# ==========================================
# METRICS
# ==========================================
def compute_stats(pred_binary, target, mask):
    """
    Accumulate TP/FP/FN/TN over pixels where mask==1.
    pred_binary, target, mask : numpy arrays [1, H, W] float32
    """
    m  = mask.squeeze().astype(bool)
    p  = pred_binary.squeeze()[m].astype(bool)
    t  = target.squeeze()[m].astype(bool)

    tp = int(( p &  t).sum())
    fp = int(( p & ~t).sum())
    fn = int((~p &  t).sum())
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def metrics_from_stats(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    fpr       = fp / (fp + tn + 1e-6)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, fpr=fpr)


# ==========================================
# SPLIT AUX
# ==========================================
def split_aux(aux_tensor):
    return [aux_tensor[:, i:i+1] for i in range(aux_tensor.shape[1])]


# ==========================================
# MAIN EVALUATION
# ==========================================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 Evaluation  |  device={device}\n")

    print("Loading checkpoints...")
    struct_enc, fusion, decoder = load_models(device)

    dataset = BanglaSlumDataset(
        data_dir=DATA_DIR,
        use_hazy=False,
        return_metadata=True
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Aggregate accumulators
    hc_tp   = hc_fp   = hc_fn   = hc_tn   = 0
    all_tp  = all_fp  = all_fn  = all_tn  = 0
    ctrl_fp = ctrl_total = 0

    per_tile_rows = []

    print(f"\nRunning on {len(dataset)} tiles...\n")

    with torch.no_grad():
        for batch in loader:
            tid = batch["tile_id"][0]

            image      = batch["image"].to(device)
            aux        = batch["aux"].to(device)
            target     = batch["target"].numpy()          # [1, 1, 512, 512]
            hc_eval    = batch["hc_eval_mask"].numpy()
            hc_slum    = batch["hc_slum_mask"].numpy()
            all_labeled= batch["all_labeled_mask"].numpy()
            hc_formal  = batch["hc_formal_mask"].numpy()

            # Forward pass
            se_stack  = split_aux(aux)
            s_m       = struct_enc(image)
            f_m       = fusion(s_m, se_stack)
            logits    = decoder(f_m)
            probs     = torch.sigmoid(logits).cpu().numpy()   # [1, 1, 512, 512]
            pred_bin  = (probs >= THRESHOLD).astype(np.float32)

            # --- Per-tile HC stats ---
            t_tp, t_fp, t_fn, t_tn = compute_stats(
                pred_bin[0], target[0], hc_eval[0]
            )
            t_metrics = metrics_from_stats(t_tp, t_fp, t_fn, t_tn)

            # --- Per-tile All stats ---
            a_tp, a_fp, a_fn, a_tn = compute_stats(
                pred_bin[0], target[0], all_labeled[0]
            )
            a_metrics = metrics_from_stats(a_tp, a_fp, a_fn, a_tn)

            # --- Formal control FPR (if applicable) ---
            is_ctrl = any(ctrl in tid.lower() for ctrl in FORMAL_CONTROL_TILES)
            t_ctrl_fp    = int((pred_bin[0].squeeze() * hc_formal[0].squeeze()).sum())
            t_ctrl_total = int(hc_formal[0].squeeze().sum())

            # Accumulate global stats
            hc_tp  += t_tp;  hc_fp  += t_fp;  hc_fn  += t_fn;  hc_tn  += t_tn
            all_tp += a_tp;  all_fp += a_fp;   all_fn += a_fn;  all_tn += a_tn
            if is_ctrl:
                ctrl_fp    += t_ctrl_fp
                ctrl_total += t_ctrl_total

            per_tile_rows.append({
                "tile_id":    tid,
                "is_control": is_ctrl,
                "hc_iou":     t_metrics["iou"],
                "hc_f1":      t_metrics["f1"],
                "hc_prec":    t_metrics["precision"],
                "hc_rec":     t_metrics["recall"],
                "all_iou":    a_metrics["iou"],
                "formal_fpr": t_ctrl_fp / (t_ctrl_total + 1e-6),
            })

            status = "🏛️  CTRL" if is_ctrl else "🏘️  SLUM"
            print(
                f"  {status}  {tid:<30}  "
                f"HC-IoU={t_metrics['iou']:.4f}  "
                f"All-IoU={a_metrics['iou']:.4f}  "
                f"P={t_metrics['precision']:.4f}  "
                f"R={t_metrics['recall']:.4f}"
            )

    # ==========================================
    # AGGREGATE RESULTS
    # ==========================================
    hc_global  = metrics_from_stats(hc_tp,  hc_fp,  hc_fn,  hc_tn)
    all_global = metrics_from_stats(all_tp, all_fp, all_fn, all_tn)
    formal_fpr = ctrl_fp / (ctrl_total + 1e-6)

    print("\n" + "=" * 65)
    print("RESULTS — BanglaSlumNet Stage 2 Evaluation")
    print("=" * 65)
    print(f"  HC-IoU   (primary)  : {hc_global['iou']:.4f}")
    print(f"  All-IoU  (secondary): {all_global['iou']:.4f}")
    print(f"  HC-F1               : {hc_global['f1']:.4f}")
    print(f"  HC-Precision        : {hc_global['precision']:.4f}")
    print(f"  HC-Recall           : {hc_global['recall']:.4f}")
    print(f"  All-Precision       : {all_global['precision']:.4f}")
    print(f"  All-Recall          : {all_global['recall']:.4f}")
    print(f"  Formal-dense FPR    : {formal_fpr:.4f}  "
          f"({ctrl_fp}/{ctrl_total} formal pixels flagged as slum)")
    print("=" * 65)

    # ==========================================
    # SAVE TO CSV
    # ==========================================
    csv_path = os.path.join(RESULTS_DIR, "eval_results.csv")
    with open(csv_path, "w") as f:
        f.write("tile_id,is_control,hc_iou,hc_f1,hc_prec,hc_rec,"
                "all_iou,formal_fpr\n")
        for row in per_tile_rows:
            f.write(
                f"{row['tile_id']},{row['is_control']},"
                f"{row['hc_iou']:.6f},{row['hc_f1']:.6f},"
                f"{row['hc_prec']:.6f},{row['hc_rec']:.6f},"
                f"{row['all_iou']:.6f},{row['formal_fpr']:.6f}\n"
            )

    # Summary row
    summary_path = os.path.join(RESULTS_DIR, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("BanglaSlumNet Stage 2 — Evaluation Summary\n")
        f.write("=" * 65 + "\n")
        f.write(f"HC-IoU        : {hc_global['iou']:.4f}\n")
        f.write(f"All-IoU       : {all_global['iou']:.4f}\n")
        f.write(f"HC-F1         : {hc_global['f1']:.4f}\n")
        f.write(f"HC-Precision  : {hc_global['precision']:.4f}\n")
        f.write(f"HC-Recall     : {hc_global['recall']:.4f}\n")
        f.write(f"All-Precision : {all_global['precision']:.4f}\n")
        f.write(f"All-Recall    : {all_global['recall']:.4f}\n")
        f.write(f"Formal FPR    : {formal_fpr:.4f}\n")
        f.write(f"Tiles evaluated: {len(per_tile_rows)}\n")
        f.write(f"Control tiles  : "
                f"{sum(1 for r in per_tile_rows if r['is_control'])}\n")

    print(f"\n📄 Per-tile CSV  : {csv_path}")
    print(f"📄 Summary       : {summary_path}")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    evaluate()