# YOLO26 at ~5 W: Hailo-8L, Hailo-8, and Jetson Orin Nano Compared

31.05.2026

After porting YOLO26 to the Hailo-8L on a Raspberry Pi 5 ([porting chronology](../yolo26n-hailo-L8/), [quantization noise analysis](../quantization_noise_analysis_yolo26/)), I ran the same model on the **Jetson Orin Nano 8GB at its 7 W power mode** (which draws 4.6-5.3 W actual, matching Pi 5 + Hailo) and on the full **Hailo-8** (26 TOPS, Tera Operations Per Second) as a third reference point. This compares all three at the same power budget.

**At matched ~5 W power on YOLO26, no single accelerator dominates.** Hailo-8 wins small-model speed across n/s/m. Jetson wins accuracy at every size, runs `l` directly from off-the-shelf Ultralytics exports while Hailo `l` requires Dataflow Compiler (DFC) self-compilation (covered in my [prior case study](../yolo26n-hailo-L8/)). Hailo-8L is the cheap-and-low-power entry point. Each platform's silicon is sized for a different part of the curve.

## Headline result

<iframe src="../images/jetson_vs_hailo_pareto.html" width="100%" height="600px" style="border:none;"></iframe>
*Figure 1: Inference latency vs mean Average Precision (mAP) across the three accelerator options and four model sizes. Top-left is the desirable corner. FP32 reference per variant shown as horizontal dotted lines.*

1. **Hailo-8 wins small-and-fast.** 84 FPS end-to-end on `n`. If your application fits `n` or `s` and FPS matters more than 0.025 mAP, this is the pick.
2. **Jetson FP16 wins accuracy at every size.** Matches FP32 to within rounding. Also the only platform that runs `l` from an off-the-shelf export.

### Full numbers

Full COCO val2017, 1000 frame sustained timing after warmup. Inference is accelerator silicon only (host-to-device copy + execute + device-to-host copy). Deployable FPS = `1000 / (Pi 5 pre + accelerator inference + Pi 5 post)` for all rows, so the host-CPU side is the same Pi 5 across platforms.

| Variant | Hailo-8L (Pi 5) | Hailo-8 (Pi 5) | Jetson FP16 (7W mode) |
|:---|:---:|:---:|:---:|
| **Inference (ms)** | | | |
| `n` | 12.09 | **9.81** | 15.58 |
| `s` | 21.68 | **17.60** | 26.76 |
| `m` | 47.98 | **34.87** | 52.88 |
| `l` | self-compile† | self-compile† | **64.20** |
| **mAP@.5:.95** | | | |
| `n` | 0.375 | 0.3748 | **0.402** |
| `s` | 0.451 | 0.4507 | **0.477** |
| `m` | 0.502 | 0.5024 | **0.524** |
| `l` | self-compile | self-compile | **0.541** |
| **Deployable FPS** | | | |
| `n` | 70.9 | **83.8** | 56.9 |
| `s` | 41.8 | **50.3** | 34.5 |
| `m` | 19.9 | **26.8** | 18.1 |
| `l` | self-compile | self-compile | **15.0** |
| **System power (W, total)** | | | |
| `n` | **3.9**‡ | 4.8‡ | 4.6 |
| `s` | **3.5**‡ | 4.6‡ | 4.8 |
| `m` | **3.3**‡ | 4.4‡ | 5.3 |
| `l` | self-compile | self-compile | 5.3 |

† Hailo Model Zoo currently ships `yolo26n/s/m` Hailo Executable Format (HEF) files for Hailo-8L and Hailo-8 but not `yolo26l`. The DFC self-compile flow is documented in the [porting case study](../yolo26n-hailo-L8/).

‡ Hailo system power = measured Pi 5 PMIC (Power Management Integrated Circuit) + Hailo chip data sheet typical (Hailo-8L 1.5 W, Hailo-8 2.5 W). Jetson is directly measured. See [Power](#power) section for the methodology asymmetry.

## The setup

The Jetson is an **AGX Orin Developer Kit reflashed as a Nano 8GB** (`jetson-agx-orin-devkit-as-nano8gb internal`, NVIDIA-supported emulation), running at the **7 W power mode** (`nvpmodel -m 1`). At 7 W the Jetson runs on 4 cores capped at 960 MHz, with the GPU at 408 MHz and only 2 of 4 Texture Processing Clusters (TPCs) active. Actual measured power draw during YOLO inference: 4.6-5.3 W. The Hailo systems land in the 3.3-4.8 W range (Hailo-8L data sheet typical 1.5 W + Pi 5; Hailo-8 data sheet typical 2.5 W + Pi 5). All three are in the same single-digit-watt envelope on this workload.

The Jetson's default power mode is **15 W** (`nvpmodel -m 0`), not 7 W. I chose 7 W here to match the Pi 5 + Hailo power envelope so the comparison is silicon-vs-silicon at the same budget, not budget-vs-budget. The 15 W comparison (where the Jetson is roughly 2× faster on `m`/`l` but draws ~7 W actual) is its own follow-up post.

Software: TensorRT 10.3, JetPack 6.2.2, CUDA 12.6. Hailo side unchanged from the [prior case study](../yolo26n-hailo-L8/): HailoRT 4.x C++ on Raspberry Pi 5 8 GB stock 2.4 GHz.

**Same C++ pre/post on both sides**, identical `preprocess`/`postprocess`/`stats` headers byte for byte. Only the inference call differs (HailoRT vs TensorRT). Differences in the tables are silicon, driver, engine, not pipeline.

**The Jetson recipe** is the ONNX (Open Neural Network Exchange) model from Ultralytics, built with `BuilderFlag.FP16`, no calibration. mAP retention 100 % vs FP32. Why FP16 not INT8: my current INT8 recipe (head FP16, backbone INT8 via TRT entropy calibration) at `m` is 17.1 ms / 0.468 mAP vs FP16's 22.8 ms / 0.524 mAP. That's 25 % faster for 0.056 mAP lost (~11 % relative). Recovering that accuracy without losing the speedup is its own post.

## Where each platform wins

Three different specialisations, driven by what the silicon was designed for. **Hailo-8** has the most compute throughput (26 TOPS) and the most on-chip SRAM (Static Random Access Memory) among the Hailo SKUs, so it splits YOLO26 into fewer execution contexts and wins raw speed on the variants it ships. **Hailo-8L** is the lowest-power option on the table (1.5 W chip typical; 3.3-3.9 W system) and is sized for small models where its limited SRAM still fits the workload acceptably. **Jetson FP16** has FP16 hardware that Hailo lacks, so it hits FP32-equivalent accuracy on every variant without any quantization recipe; at the 7 W mode its compute is throttled enough to give up the raw-speed advantage to Hailo at smaller variants, but it remains the only path to an off-the-shelf `l`.

The architectural reason these specialisations exist is the next section.


## Architectural finding: per-context overhead dominates Hailo latency at batch 1

The question: why is yolo26n on Hailo-8L 12 ms when 5.4 GFLOPs ÷ 13 TOPS = 0.4 ms theoretical? Compute can't be the dominant cost.

**Hypothesis.** The Hailo Dataflow Compiler partitions a model across multiple **execution contexts** because the model doesn't fit in on-chip SRAM. Each context is loaded into SRAM via PCIe DMA (Direct Memory Access), the LCU (compute) sequencer fires, output is moved out, then the next context loads. At batch 1, this load-execute-unload cycle runs serially per context per frame, and the per-context overhead (DMA + control plane) doesn't shrink with model size.

**Direct test via `hailortcli run2 measure-fw-actions`.** This subcommand captures firmware-level events with clock-cycle timestamps during inference. The JSON reports `clock_cycle_MHz: 200` and Hailo's community forum confirms 200 MHz is the NPU (Neural Processing Unit) compute clock on PCIe-constrained Hailo-8 variants (which the Hailo-8L on Pi 5 is). I treat that as the timestamp clock; the empirical cross-check is that sum-of-spans converted at 200 MHz matches HW Latency from `hailortcli benchmark` within ~4 %. Each context's events include `trigger_sequencer` and `sequencer_done_interrupt` brackets, which identify LCU compute-active time. Everything else in the action list (DMA setup, channel activation, interrupts, inter-context buffer hookup) is the overhead bucket.

```bash
hailortcli run2 --mode raw_async -t 10 set-net yolo26m.hef \
                measure-fw-actions --output-path runtime_data.json
```

The JSON output has per-context action lists. Sum of (`sequencer_done_interrupt.timestamp` − `trigger_sequencer.timestamp`) for each compute bracket gives LCU active time per context. Span (last − first action timestamp) gives total per-context time. Difference is the overhead.

**Results for two variants:**

| Variant | HW Latency | Sum of context spans | LCU compute | DMA + control overhead |
|:---|:---:|:---:|:---:|:---:|
| yolo26n | 12.18 ms | 12.72 ms (✓ matches) | 4.93 ms (**39 %**) | 7.80 ms (**61 %**) |
| yolo26m | 46.25 ms | 48.13 ms (✓ matches) | 9.97 ms (**21 %**) | 38.16 ms (**79 %**) |

For yolo26n, almost two-thirds of HW latency is non-compute. For yolo26m, four-fifths. The sum-of-spans matches HW Latency from `hailortcli benchmark` within 1-4 %, which cross-validates the firmware-level measurement.

**Validation via batch sweep.** If per-context overhead is real and per-batch-amortizable, raising the batch size should reduce per-frame chip time. Running `hailortcli benchmark --batch-size N`:

| Variant | batch=1 per-frame | batch=4 per-frame | batch=8 per-frame | Reduction at batch=8 |
|:---|:---:|:---:|:---:|:---:|
| yolo26n | 12.78 ms | 8.16 ms | 7.39 ms | **−42 %** |
| yolo26m | 48.4 ms | 30.8 ms | 28.7 ms | **−41 %** |

At batch 8, per-frame chip time drops ~40 % for both variants. That's the amortization in action: ~40 % of the batch=1 chip time was per-batch fixed cost that gets spread across the batch.

**What this means.** Hailo's silicon is designed to be efficient at moderate-to-high batch sizes. The architectural tradeoff is real and explicit in their documentation: more contexts means more DMA-loadable model, at the cost of per-frame overhead that only amortizes at batch > 1. For edge deployments running at batch 1, you pay the full overhead, and the larger the model, the more contexts it splits across, the higher the overhead share.

This is not a deficiency. It's a design choice that optimises for the workloads Hailo's customers ship most: high-throughput batched inference in a smart camera or an automotive perception stack. The Pi 5 + Hailo-8L edge use case (batch 1, latency-sensitive) is the opposite extreme and exposes the cost.

**Limits of this measurement.** The "overhead" bucket includes DMA + control + setup but doesn't separate them. The firmware actions expose `descriptors_count` for some DMA fetches but not actual byte counts, so I can't directly attribute overhead between weight DMA, activation DMA, and control plane. `--mode raw_async` is required for `measure-fw-actions`, which bypasses data transformations, so the per-context spans I measured are pure chip time, not full streaming. And these are batch 1 measurements; at batch > 1 the per-context cost is shared across the batch and the breakdown changes.

## Power

Data sheet typicals: Hailo-8L 1.5 W, Hailo-8 2.5 W. Pi 5 idle ~2.7 W (rising under inference load). Pi 5 + Hailo-8L total: ~4-5 W sustained.

Jetson measured power during YOLO inference at 7 W mode: **4.6-5.3 W total** (via tegrastats sum), close to the Pi 5 + Hailo envelope. The "7 W" nameplate is a configured cap; actual draw at batch 1 is lower than the cap on the smaller variants and approaches the cap on `m`/`l`.

Apples-to-apples power comparison has caveats: Pi 5's PMIC doesn't capture the Hailo's draw (the AI Kit HAT, Hardware Attached on Top, regulates 3.3 V locally from 5 V via PCIe FFC, Flexible Flat Cable, bypassing PMIC rails), and the Jetson's `VDD_GPU_SOC` rail covers more than just the GPU. A USB-C wall meter at each platform's input would give the definitive whole-platform answer.

## Methodology summary

Every number is from a sustained 1000-frame benchmark after 50 warmup frames. mAP via pycocotools COCOeval on full COCO val2017 (5000 images, Intersection over Union, IoU, 0.50:0.95, conf 0.001).

- **Hailo:** HailoRT 4.x C++ + Hailo Model Zoo v2.18.0 HEFs on Raspberry Pi 5 8 GB stock 2.4 GHz.
- **Jetson:** TensorRT 10.3, JetPack 6.2.2 (CUDA 12.6, cuDNN 9.3), 7 W power mode (`nvpmodel -m 1`). Ultralytics ONNX, `BuilderFlag.FP16`, no calibration. Same C++ pre/post headers as the Hailo side.
- **Power:** Jetson via tegrastats sum-of-rails. Hailo system power = Pi 5 PMIC + Hailo chip data sheet typical (see [Power](#power) section for the methodology asymmetry).
- **Firmware profiling for the architectural section:** `hailortcli run2 --mode raw_async -t 10 set-net <hef> measure-fw-actions --output-path runtime_data.json` produces per-context action lists with cycle timestamps. Parsed in Python: LCU compute = sum of `(sequencer_done_interrupt − trigger_sequencer)` cycle deltas per context; overhead = span minus LCU compute. Reproducible from any HailoRT install + a HEF.

If you're working on a similar problem, [get in touch](../contact.md). Happy to share the full pipeline, source-of-truth JSONs, and the firmware-profiling parse script.


## Open questions

1. **Can a Jetson INT8 recipe beat FP16?** INT8 has a 2x Tensor Core edge over FP16 that early experiments suggest is partly recoverable. First attempts land slower or less accurate than FP16. Next: per-layer SQNR (Signal-to-Quantization-Noise Ratio) analysis to find which Conv layers need higher precision, combined with `nvidia-modelopt` autotune. Possible follow-up post.
2. **Real Nano vs AGX-as-Nano emulation.** Direct measurement on a retail Nano dev kit would close the emulation caveat.

## Conclusion

Two methodology lessons that came out of this work.

**Power-matched comparison matters.** "Hailo at 5 W vs Jetson at 15 W" would have made the Jetson look roughly 2× faster at `m` than it actually is at matched 5 W power. The vendor nameplates ("15 W Thermal Design Power, TDP" vs "1.5 W typical chip") don't reflect actual draw at batch-1 edge inference, which lands in 3-5 W for all three platforms tested. Run the comparison at the configuration you'd actually deploy.

**Inside-the-silicon profiling matters.** The FPS-per-TOPS marketing view of Hailo undersells how much of its HW latency is per-context overhead at batch 1 (60-80 % depending on model size). The Hailo Dataflow Compiler's per-context cost only amortizes at batch > 1; for batch-1 edge deployment you pay it in full. Firmware-level profiling via `hailortcli run2 measure-fw-actions` exposed this directly; the FPS benchmark alone didn't.

Next in this series: [Jetson Orin Nano 7 W vs 15 W on YOLO26: same idle floor, half the headroom](../yolo26-jetson-power-modes/). At 15 W (the Jetson default), the silicon delivers roughly 2× the throughput on `m`/`l` while drawing 5-7 W actual. The follow-up unpacks where that extra performance comes from architecturally and shows that the "lower power mode = more efficient" intuition is wrong on this silicon.

What this post doesn't settle: real Nano vs AGX-as-Nano emulation, thermal-constrained sustained throughput, and at-the-wall power measurement on the Hailo side.
