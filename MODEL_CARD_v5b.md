# ReSnap model v5b

YOLOv8s, 30 ingredient classes, trained after the course project (the course report and slides use v4b).
`models/best_v5b.pt` (and `best_v5b.onnx`) use the same classes, class order and output format as v4b, so the model is a drop-in replacement for `models/best.pt`.

## What changed from v4b

| Run | Single change | Result |
|---|---|---|
| v5a | +1,702 training photos of ginger, turmeric, galangal (Roboflow: soda, rimpang10, ayush), including 200 unlabeled negatives (kencur, temulawak) | Accepted. Better on rhizomes, but false detections on photos of other spices doubled (garlic/egg on peppercorn piles). |
| v5b | v5a + 1,038 photos from the Roboflow "Deteksi Rempah" (relz) dataset: garlic, shallot, onion, ginger, galangal labeled; other spices (pepper, cloves, cardamom, nutmeg, fennel, ...) left unlabeled as negatives | Chosen as the v5 model, with one recorded exception (below). |

The training recipe is identical to v4b (YOLOv8s from COCO weights, 640 px, AdamW lr0 0.0003, weight decay 0.001, mosaic 1.0, mixup 0.1, 70 epochs, patience 15). Every image added was checked with dHash (including mirrored) against all ReSnap splits and the external test source, and photos where v4b found other unlabeled ReSnap ingredients were removed.

## Results

Old test set (654 photos, same one used for v4b in the course report). Evaluated once, after v5b was chosen.

| | v4b | v5b |
|---|---|---|
| Precision | 0.684 | 0.735 |
| Recall | 0.650 | 0.643 |
| F1 | 0.667 | 0.686 |
| mAP@0.5 | 0.694 | 0.711 |
| mAP@0.5:0.95 | 0.465 | 0.470 |

External rhizome test (314 photos from a different Roboflow source, `rival`, never used for training or model selection; 202 of them show other spices only):

| | v4b | v5a | v5b |
|---|---|---|---|
| mAP@0.5 (ginger, turmeric, galangal) | 0.415 | 0.493 | 0.595 |
| AP50 galangal / turmeric / ginger | 0.43 / 0.17 / 0.64 | 0.61 / 0.22 / 0.65 | 0.85 / 0.27 / 0.67 |
| Spice photos with a false detection (conf ≥ 0.5) | 9.9% | 20.8% | 6.4% |

Validation set (1,069 photos): mAP@0.5 0.694 (v4b) → 0.713 (v5b), mAP@0.5:0.95 0.481 → 0.497, train-val gap 0.274 → 0.260.

## Selection exception

Acceptance rules were written before each training run. v5b passed 4 of 5. It failed "Open Images holdout AP50 must not drop more than 5 points vs v5a": mushroom dropped 6.5 points (76 objects) and potato 20 points (13 objects; v5b is still slightly above v4b on potato). v5b was chosen anyway because it was better than v5a on validation mAP, false detections on spice photos, rhizomes and overfitting gap. This decision was made before the test sets were evaluated.

## Known weaknesses

- Broccoli is weaker than v4b on the old test set (AP50 0.37 → 0.26); shallot and lemongrass drop about 8 points. Broccoli suffers from pile annotations and medium-size objects.
- Turmeric stays weak on both test sets (AP50 0.22 on the old test set, 0.27 on the external test), although validation improved to 0.48.
- Several training/validation images annotate a pile of items with one box, which lowers measured scores for broccoli, mushroom and cauliflower.

## Data sources added in v5

All CC BY 4.0, from Roboflow Universe:
[soda-fj5ov/deteksi-rempah-rempah-2](https://universe.roboflow.com/soda-fj5ov/deteksi-rempah-rempah-2),
[rimpang-key8c/rempah-c9tpp](https://universe.roboflow.com/rimpang-key8c/rempah-c9tpp),
[ayushvision/ayushvision_tuber](https://universe.roboflow.com/ayushvision/ayushvision_tuber),
[relz-f3q8v/deteksi-rempah-jjceg](https://universe.roboflow.com/relz-f3q8v/deteksi-rempah-jjceg);
external test: [rival-0zxtv/rempah-r4adj](https://universe.roboflow.com/rival-0zxtv/rempah-r4adj).

Notebooks: `notebooks/ReSnap_Analisis_Error_V4b_Kaggle.ipynb` → `ReSnap_V5_Audit_Tumpukan_Rimpang_Kaggle.ipynb` → `ReSnap_V5a_Data_Train_Kaggle.ipynb` → `ReSnap_V5a_Test_Eksternal_Kaggle.ipynb` → `ReSnap_V5b_Rempah_Relz_Kaggle.ipynb` → `ReSnap_V5b_Evaluasi_Final_Kaggle.ipynb`. Result files are in `hasil/v5/`.
