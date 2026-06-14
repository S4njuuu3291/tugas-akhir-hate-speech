# Dokumentasi Config — `config.yaml`

Semua konfigurasi di-load di `main.py` lewat `CONFIG = yaml.safe_load(f)`.

## `paths`
| Key | Default | Fungsi |
|---|---|---|
| `kamus_alay` | `DATA/raw/kamus_alay_normal.csv` | Path kamus slang → baku |
| `model_bert` | `MODELS/model_hs_indobert` | Path model IndoBERT |
| `artifact_baseline` | `MODELS/baseline/model_artifact_2.pkl` | Path pickle artifact (vectorizer, model klasik) |

## `preprocessing`
| Key | Default | Fungsi |
|---|---|---|
| `max_words` | `40` | Maks token teks input |

## `model.bert`
| Key | Default | Fungsi |
|---|---|---|
| `max_length` | `128` | Maks panjang token BERT |
| `thresholds` | `[0.16, 0.64, 0.18, 0.66, 0.12, 0.34]` | Threshold per label (6 label) |

## `ui`
| Key | Default | Fungsi |
|---|---|---|
| `text_area_height` | `150` | Tinggi input teks (px) |
| `shap_samples.min` | `64` | Min sampel KernelSHAP |
| `shap_samples.max` | `2048` | Max sampel KernelSHAP |
| `shap_samples.default` | `256` | Default sampel KernelSHAP |
| `shap_samples.step` | `64` | Step sampel KernelSHAP |
