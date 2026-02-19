# Quantization Noise Analysis: YOLO26 on Hailo-8L
19.02.2026

This post documents a layer-by-layer noise analysis of the YOLO26 model family after INT8 quantization on the Hailo-8L, motivated by an anomalous accuracy drop in the Medium variant. The analysis was performed using the Hailo Dataflow Compiler's `analyze_noise` tool. For context on the porting process and hybrid architecture, see the [full porting chronology](../yolo26n-hailo-L8/).

## 1. Accuracy Retention by Variant

Post-quantization benchmarks across the full YOLO26 family produced a non-monotonic accuracy retention curve:

<iframe src="../images/plot_a_quantization_cliff_interactive.html" width="100%" height="600px" style="border:none;"></iframe>
*Figure 1: FP32 vs. INT8 mAP across variants. YOLO26-M shows disproportionate accuracy loss.*

| Model | CPU mAP (FP32) | Hailo mAP (INT8) | Accuracy Retention |
|:------|:--------------:|:-----------------:|:-----------------:|
| yolo26n | 0.402 | 0.371 | 92.3% |
| yolo26s | 0.477 | 0.424 | 88.9% |
| yolo26m | 0.525 | 0.441 | 84.0% |
| yolo26l | 0.541 | 0.473 | 87.4% |

YOLO26-M has the lowest accuracy retention (84.0%) despite having a higher FP32 baseline than Small. YOLO26-L, which has more layers below the 10 dB SNR threshold in absolute terms (271 vs. 158), recovers to 87.4%, suggesting its parameter count provides sufficient redundancy to absorb quantization noise that M cannot.

## 2. Layer-by-Layer SNR Analysis

The SNR analysis below shows per-layer signal fidelity across all four variants. The standard threshold for 8-bit quantization is **10 dB**; below this, quantization error is large enough to materially degrade layer output.

<iframe src="../images/plot_b_heartbeat_noise_interactive.html" width="100%" height="600px" style="border:none;"></iframe>
*Figure 2: Per-layer SNR. YOLO26-M drops below 10 dB in early layers and does not recover.*

| Variant | Total Layers | Min SNR (dB) | Mean SNR (dB) | Layers < 10 dB |
|:--------|:------:|:------------:|:-------------:|:--------------:|
| yolo26n | 174 | 14.77 | 18.15 | 0 |
| yolo26s | 174 | 12.19 | 17.46 | 0 |
| yolo26m | 188 | 9.74 | 15.42 | 158 |
| yolo26l | 275 | 9.64 | 15.76 | 271 |

YOLO26-N and YOLO26-S have zero layers below threshold. YOLO26-M has 158/188 layers below 10 dB — noise is pervasive, not isolated. The analysis identified three architectural sources: early convolutions, feature splitters, and attention mechanisms.

### 2.1 Early Convolution Noise: `conv2`

`conv2` appears in the worst-10 layers for YOLO26-S, M, and L. The cause is an increasingly peaked activation distribution as model width increases:

| Variant | Activation Concentration (top 1% bins) | Activation Utilization | Min SNR |
|:--------|:--------------------------------------:|:----------------------:|:-------:|
| yolo26n | 46% | 0.36 | 15.2 dB |
| yolo26s | 58% | 0.32 | 12.4 dB |
| yolo26m | **62%** | **0.18** | 9.8 dB |
| yolo26l | 60% | 0.22 | 9.7 dB |

In YOLO26-M, 62% of activations fall in the top 1% of histogram bins, forcing a wide clip range (~40). Only 18% of the INT8 dynamic range covers the high-density region. Because `conv2` is early in the network, this error propagates and compounds through downstream layers.

### 2.2 Feature Splitter Layers

`conv_feature_splitter` layers appear in the worst-10 for YOLO26-S, M, and L. These layers partition large channel tensors for efficient processing. In wider models, the partitions develop large weight asymmetry: one partition may contain large weights while another contains near-zero values. A single INT8 scale factor must cover the full tensor, causing small weights to collapse to zero.

| Layer | Variant | Weight Range | Weight Asymmetry | SNR |
|:------|:--------|:-----------:|:----------------:|:---:|
| conv_feature_splitter2_1 | yolo26s | 3.22 | 0.309 | 12.5 dB |
| conv_feature_splitter10_2 | yolo26m | 1.08 | 0.222 | 9.7 dB |
| conv_feature_splitter1_2 | yolo26m | 4.33 | 0.494 | 9.8 dB |
| conv_feature_splitter1_2 | yolo26l | 6.87 | 0.637 | 9.6 dB |

Weight asymmetry is on a 0–1 scale (0 = symmetric, 1 = fully one-sided). YOLO26-S peaks at 0.309; YOLO26-M reaches 0.494; YOLO26-L reaches 0.637 with a weight range of 6.87. At an asymmetry of ~0.5, roughly half of the INT8 bins cover the near-empty side of the distribution, wasting a significant portion of the available precision.

### 2.3 Attention Layers

`matmul3` and `ew_sub_softmax2` appear in YOLO26-M's worst-10. Softmax outputs are concentrated near 0 and 1, making them poorly suited to uniform INT8 quantization.

The activation histogram for `ew_sub_softmax2` — the worst-performing layer in YOLO26-M — illustrates this directly:

<iframe src="../images/plot_c_wasted_bits_interactive.html" width="100%" height="600px" style="border:none;"></iframe>
*Figure 3: INT8 bin occupancy for `ew_sub_softmax2`. 84% of bins receive zero activations.*

This layer uses only **41 out of 256** INT8 bins. The softmax output is bimodal (concentrated near 0 and 1), but the uniform quantization grid allocates bins evenly across [0, 1], wasting the entire midrange. The downstream regression branch (`output_layer5`) drops to 9.7 dB as a result — the attention mechanism lacks the precision to distinguish between activation levels in the relevant range.

## 3. Regression vs. Classification Output Sensitivity

The six output heads split into three regression heads (`output_layer1,3,5` — 4 bounding-box coordinates) and three classification heads (`output_layer2,4,6` — 80-class logits). Per-head SNR across all variants:

| Head | Type | yolo26n | yolo26s | yolo26m | yolo26l |
|:-----|:----:|:-------:|:-------:|:-------:|:-------:|
| output_layer1 | reg | 17.08 dB | 13.48 dB | 11.04 dB | 9.69 dB |
| output_layer2 | cls | 26.59 dB | 23.27 dB | 24.06 dB | 22.20 dB |
| output_layer3 | reg | 15.41 dB | 13.38 dB | 9.94 dB | 9.64 dB |
| output_layer4 | cls | 17.77 dB | 20.69 dB | 18.81 dB | 20.92 dB |
| output_layer5 | reg | 15.16 dB | 12.19 dB | 9.74 dB | 10.32 dB |
| output_layer6 | cls | 14.77 dB | 18.97 dB | 17.17 dB | 20.14 dB |
| **mean reg** | | **15.88 dB** | **13.02 dB** | **10.24 dB** | **9.88 dB** |
| **mean cls** | | **19.71 dB** | **20.98 dB** | **20.01 dB** | **21.09 dB** |

Contrary to the intuition that classification is harder to quantize, regression heads are consistently and significantly noisier. The cls/reg SNR gap widens with model size: ~4 dB in Nano, ~8 dB in Small, ~10 dB in Medium and Large. The worst-case head for YOLO26-N is `output_layer6` (cls, 14.77 dB), but for every other variant it is a regression head — `output_layer5` for YOLO26-S (12.19 dB) and YOLO26-M (9.74 dB), and `output_layer3` for YOLO26-L (9.64 dB).

Regression and classification heads share most of the backbone ancestors, including all attention layers, so there is no separate architectural path. The SNR gap arises at the output heads themselves: each regression head compresses 64 backbone channels down to 4 coordinate values (16x reduction), while each classification head compresses 256 channels to 80 logits (3.2x reduction). With only 4 output dimensions, the accumulated upstream noise has nowhere to average out, making the regression outputs structurally more sensitive to quantization noise regardless of which backbone layer is the source.

## Conclusion & Path Forward

Standard uniform INT8 quantization is insufficient for YOLO26-M on the Hailo-8L. The noise is systemic (158/188 layers below 10 dB), originating from three identifiable layer classes: early convolutions with peaked activation distributions, feature splitters with high weight asymmetry, and attention layers with bimodal softmax outputs.

Post-quantization fine-tuning (Hailo DFC optimization level 3) did not meaningfully improve accuracy — YOLO26-M recovered less than 0.5% mAP, consistent with the noise being structural rather than a calibration issue.

The most straightforward mitigation is **mixed precision** on the bottleneck layers, which is expected to carry a latency cost; the tradeoff needs measurement.

Additional candidates to investigate:

- **Percentile-based activation clipping** on `conv2` and feature splitter layers to reduce the effective clip range and improve INT8 bin utilization
- **Deeper regression heads** with more intermediate layers between the backbone and the 4-coordinate output, spreading the compression gradually (e.g., 64 → 32 → 16 → 4) rather than collapsing in one step. This requires retraining the model

None of these have been validated yet. The noise analysis provides the layer-level targets; the next step is to measure the accuracy and latency impact of each intervention.
