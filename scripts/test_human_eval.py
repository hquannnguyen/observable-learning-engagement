import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scripts.evaluate_m0_heuristic import assess_tensor_window
from src.uncertainty.gate import UncertaintyThresholds, explain_uncertainty_gate
from src.contracts.labels import CANONICAL_TRAINING_LABELS, INDEX_TO_LABEL, UNCERTAIN_LABEL
from src.models.factory import model_from_checkpoint
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

ann = pd.read_csv('data/annotations/review_v03_independent_20261124/annotator_b.csv')
test_ann = ann[ann['split'] == 'test'].copy()
valid_test = test_ann[test_ann['annotation_state'] == 'VALID'].copy()

print(f"=== TỔNG QUAN TẬP TEST NGƯỜI THẬT (PHONG) ===")
print(f"Tổng số cửa sổ test: {len(test_ann)}")
print(f"Số cửa sổ VALID: {len(valid_test)} ({len(valid_test)/len(test_ann)*100:.1f}%)")
print(f"Số cửa sổ OCCLUDED: {len(test_ann) - len(valid_test)} ({(len(test_ann) - len(valid_test))/len(test_ann)*100:.1f}%)")
print("Phân bổ nhãn người thật:")
for lbl, cnt in valid_test['participation_label'].value_counts().items():
    print(f"  - {lbl}: {cnt} ({cnt/len(valid_test)*100:.1f}%)")
print()

# NPZ features
npz = np.load('data/features/window_features.npz')
wids = list(npz['window_ids'])
wid_to_idx = {wid: i for i, wid in enumerate(wids)}
indices = [wid_to_idx[wid] for wid in valid_test['window_id']]
X = npz['all_features'][indices] # (276, 40, 94)
y_true = valid_test['participation_label'].values
phases = valid_test['class_phase'].fillna('lecture').values

thresholds = UncertaintyThresholds()

# -------------------------------------------------------------
# 1. BASELINE M0 (Heuristic Rule-Based)
# -------------------------------------------------------------
print("=============================================================")
print("1. ĐÁNH GIÁ BASELINE M0 (Heuristic Rule-Based - Task T20)")
print("=============================================================")
m0_preds = []
m0_accepted = []
for i in range(len(valid_test)):
    res = assess_tensor_window(X[i], phases[i], thresholds)
    m0_preds.append(res['pred_label'])
    m0_accepted.append(res['accepted'])

print('Phân bổ dự đoán của M0:')
print(pd.Series(m0_preds).value_counts().to_dict())
acc_count = sum(m0_accepted)
print(f'M0 Chấp nhận (Accepted): {acc_count}/{len(m0_accepted)} ({acc_count/len(m0_accepted)*100:.1f}%)')
print(f'M0 Bị loại vào UNCERTAIN: {len(m0_accepted) - acc_count}')

acc_idx = [i for i, a in enumerate(m0_accepted) if a]
if acc_idx:
    y_t = [y_true[i] for i in acc_idx]
    y_p = [m0_preds[i] for i in acc_idx]
    print(f"-> Độ chính xác (Accuracy): {accuracy_score(y_t, y_p)*100:.2f}%")
    print(f"-> Balanced Accuracy: {balanced_accuracy_score(y_t, y_p)*100:.2f}%")
    print(f"-> Macro F1: {f1_score(y_t, y_p, average='macro', zero_division=0):.4f}")
    print("\nChi tiết Precision / Recall / F1 trên các mẫu được chấp nhận:")
    print(classification_report(y_t, y_p, zero_division=0))
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_t, y_p, labels=['HIGH_OBSERVED_PARTICIPATION', 'LOW_OBSERVED_PARTICIPATION', 'MODERATE_OBSERVED_PARTICIPATION']))
print()

# -------------------------------------------------------------
# 2. MÔ HÌNH HỌC SÂU (LSTM Baseline)
# -------------------------------------------------------------
print("=============================================================")
print("2. ĐÁNH GIÁ MÔ HÌNH LSTM BASELINE (Pre-trained)")
print("=============================================================")
ckpt_path = Path('models/checkpoints/20261025-102826-lstm-posefeat-v2.pt')
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location='cpu')
    lstm_model = model_from_checkpoint(ckpt)
    
    with torch.no_grad():
        t_X = torch.from_numpy(X).float()
        logits = lstm_model(t_X)
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        pred_labels = [INDEX_TO_LABEL[idx] for idx in preds]
        
    print(f"-> Độ chính xác thô (Raw Accuracy): {accuracy_score(y_true, pred_labels)*100:.2f}%")
    print(f"-> Balanced Accuracy: {balanced_accuracy_score(y_true, pred_labels)*100:.2f}%")
    print(f"-> Macro F1: {f1_score(y_true, pred_labels, average='macro', zero_division=0):.4f}")
    print("\nChi tiết Classification Report:")
    print(classification_report(y_true, pred_labels, zero_division=0))
    print("Ma trận nhầm lẫn:")
    print(confusion_matrix(y_true, pred_labels, labels=['HIGH_OBSERVED_PARTICIPATION', 'LOW_OBSERVED_PARTICIPATION', 'MODERATE_OBSERVED_PARTICIPATION']))
