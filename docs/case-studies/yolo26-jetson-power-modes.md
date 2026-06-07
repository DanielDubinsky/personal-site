# Jetson Orin Nano on YOLO26: the power mode is a cap, not a budget

02.06.2026

Following the [matched-power Hailo vs Jetson comparison](../yolo26-jetson-vs-hailo/), the natural next question: 7 W or 15 W mode? The intuition is that the mode name tells you what the chip will draw, so you pick the mode that matches your power budget.

**On YOLO26 at batch 1, 15 W mode did the job within a 7 W budget.** Across all variants, 15 W mode silicon draw stayed at 4.9-7.0 W, never approaching its nameplate cap, and delivered 10-110 % more FPS than 7 W mode. 7 W mode's compute throttle was paying for a cap this workload rarely approached. That doesn't mean "use 15 W mode": your workload may differ. It means the mode name does not predict what your workload draws, so before picking a mode by which nameplate matches your budget, measure your workload's draw in each.

## Headline result

<iframe src="../images/jetson_power_modes_c3_fpsw.html" width="100%" height="500px" style="border:none;"></iframe>
*Figure 1: FPS per watt for YOLO26 n/s/m/l in both power modes (end-to-end, silicon power = VDD_GPU_SOC + VDD_CPU_CV).*

1. **On this workload, 15 W mode delivers 38-59 % higher FPS/W on `s`/`m`/`l`.** `m` goes from 3.2 to 5.0 FPS/W, `l` from 2.7 to 4.3 FPS/W.
2. **`n` is the exception, tied at 9.9 vs 10.3 FPS/W.** At 15 W mode, `n` doesn't fully load the GPU to begin with, so capping has little to cut.

### Full numbers

| Variant | 15 W mode | 7 W mode |
|:---|:---:|:---:|
| **Inference latency (end-to-end, ms)** | | |
| `n` | 13.92 | 15.58 |
| `s` | 14.84 | 26.76 |
| `m` | 22.78 | 52.88 |
| `l` | 27.58 | 64.20 |
| **FPS (end-to-end)** | | |
| `n` | **50.3** | 45.7 |
| `s` | **48.1** | 30.4 |
| `m` | **34.6** | 16.9 |
| `l` | **29.8** | 14.2 |
| **Silicon power (W, under load)** | | |
| `n` | 4.9 | 4.6 |
| `s` | 5.5 | 4.8 |
| `m` | 6.9 | 5.3 |
| `l` | 7.0 | 5.3 |
| **FPS per watt** | | |
| `n` | 10.3 | 9.9 |
| `s` | **8.7** | 6.3 |
| `m` | **5.0** | 3.2 |
| `l` | **4.3** | 2.7 |

mAP (mean Average Precision) is unchanged across modes (same FP16 engine, same model bits). 7 W mode is purely a power-cap configuration; it does not touch numerics.

## Both modes fit a 7 W envelope; only 7 W mode throttles compute

The "15 W mode" name implies the chip will draw up to 15 W. For YOLO26 at batch 1, it stays at 4.9-7.0 W across all variants. 7 W mode draws less by enforcing the cap via compute throttle, half the Texture Processing Cluster (TPC) count and capped clocks. The throttle takes effect regardless of whether the workload was approaching the cap, and on this workload it generally was not.

<iframe src="../images/jetson_power_modes_actual_vs_cap.html" width="100%" height="460px" style="border:none;"></iframe>
*Figure 2: Sustained silicon power per variant per mode under YOLO26 inference. Both modes' draws fit within a 7 W envelope. The 15 W cap is never approached.*

Both modes' silicon draw fits within a 7 W envelope for this workload. 15 W mode draws 0.3-1.7 W more than 7 W mode does, but delivers 10-110 % more FPS depending on variant (`n` +10 %, `s` +58 %, `m` +105 %, `l` +110 %). The 7 W mode's throttle is enforcing a cap that the workload rarely approached. The throughput cost is what the throttle takes; the cap is what the throttle is meant to prevent. Here, only `l` came near the cap; everything smaller had ample headroom.

**Caveat**: these are sustained 1000-frame averages. Instantaneous peaks may briefly exceed the mean. For deployments with a hard PMIC (Power Management Integrated Circuit) trip at exactly 7 W, verify with your specific workload. For thermal or average-power budgets, the sustained number is the right metric.

## What the two modes actually do

From `/etc/nvpmodel.conf` on this device (an AGX Orin Devkit reflashed as Nano 8 GB; the nvpmodel definitions match standard Nano 8 GB JetPack, see NVIDIA's [Jetson Linux Developer Guide → Platform Power and Performance](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html) for the official table):

| Knob | 15 W mode (id 0) | 7 W mode (id 1) |
|:---|:---:|:---:|
| CPU cores online | **6** | **4** |
| CPU max frequency | unrestricted (silicon max) | **960 MHz** |
| GPU TPCs enabled | **4** (`TPC_PG_MASK=240`) | **2** (`TPC_PG_MASK=252`) |
| GPU max frequency | unrestricted (silicon max) | **408 MHz** |
| EMC (External Memory Controller) max frequency | unrestricted | **2133 MHz** |

7 W mode does three things to GPU compute: halves the TPC count (raw throughput cut by half), caps the GPU clock (per-TPC throughput cut further), and caps the EMC clock (memory bandwidth cut). Combined, expect silicon compute throughput to drop ~2-2.5×. The next section measures this.

## The mechanism: same floor, more headroom

First, the idle floor in each mode. tegrastats sampled at 1 Hz for 20 seconds, system fully idle, in each mode separately (with a reboot between modes since `nvpmodel` on Orin Nano requires reboot to switch).

| Mode | VDD_GPU_SOC (idle) | VDD_CPU_CV (idle) | Silicon idle floor |
|:---|:---:|:---:|:---:|
| 15 W | 2.395 W | 0.399 W | **2.794 W** |
| 7 W  | 2.396 W | 0.399 W | **2.795 W** |

Identical to the milliwatt. `nvpmodel` caps the upper bound on power; it does not lower the floor. The System-on-Chip (SoC) fabric, memory controllers, and the GPU's bottom Dynamic Voltage and Frequency Scaling (DVFS) state are set by the silicon, not by the power mode.

7 W mode halves the GPU's TPC count and lowers the GPU clock cap to 408 MHz. The combined effect is a ~2.3× silicon slowdown, uniform across variants (2.13× for `n` through 2.41× for `m`, measured via `trtexec` GPU Compute Time). End-to-end slowdown varies: `n` only slows 1.12× because it doesn't fully load the GPU at 15 W in the first place, while `m` and `l` slow ~2.3× one-for-one because silicon dominates their latency. The power floor doesn't change either way.

You pay 2.79 W just for the chip to exist regardless of which mode you choose. The difference between modes is how much compute power you're allowed above the floor.

<iframe src="../images/jetson_power_modes_c1_divergent.html" width="100%" height="460px" style="border:none;"></iframe>
*Figure 3: Going 15 W to 7 W per variant: FPS lost (red, left) vs power saved (green, right). For every variant bigger than `n`, the FPS give-up dwarfs the power save. m and l: ~50 % FPS gone for ~25 % power saved. The asymmetry is the inefficiency.*

The headroom math, using `l` as the bottleneck case:

| Mode | Silicon floor | Silicon under `l` load | Useful compute headroom |
|:---|:---:|:---:|:---:|
| 15 W | 2.79 W | 7.0 W | **4.2 W** |
| 7 W  | 2.79 W | 5.3 W | **2.5 W** |

Same floor. 7 W mode gives you 40 % less compute headroom and delivers 52 % less FPS. The FPS drop runs ahead of the power saving because TPCs are an all-or-nothing resource: halving them halves throughput regardless of how the remaining ones are clocked.

## When measuring matters most

The "measure beyond your budget" methodology pays off when:

1. **Your model fits comfortably below the higher cap.** For batch-1 YOLO26 at any variant, both modes' actual silicon draw fits within 7 W. Skipping 15 W mode evaluation under a 7 W budget leaves 10-110 % FPS on the table at this workload.
2. **Your budget is a hard cap, not a soft preference.** If your deployment is constrained to "up to 7 W" for thermal or battery reasons, treating the 7 W mode as the only option leaves performance on the table when 15 W mode fits.
3. **You're at low batch sizes.** Smaller batches mean lower duty cycle and lower sustained draw, so the cap is least likely to engage. Larger batches push silicon higher and the cap genuinely matters.

When measurement is unlikely to change the answer:

1. **Large models or large batches that saturate 15 W mode.** If your workload at 15 W mode draws close to 15 W, the cap is enforcing a real constraint and the higher-cap mode would actually exceed a 7 W budget. Measurement confirms the obvious.
2. **Concurrent GPU workloads.** Combined draw can exceed a single workload's measurement; budget against the total, not the YOLO26 portion alone.
3. **Hard PMIC trips with no tolerance.** If your platform's power monitor will halt at exactly 7 W, instantaneous peaks above the mean matter. Either characterise peaks rigorously, or pick the safer cap.

## Methodology summary

- **Inference (end-to-end):** the same C++ bench harness used in the [matched-power case study](../yolo26-jetson-vs-hailo/). 1000 frames sustained timing after 50 warmup, TensorRT Inference column. JetPack 6.2.2, TensorRT 10.3, CUDA 12.6.
- **Inference (silicon only):** `/usr/src/tensorrt/bin/trtexec --loadEngine=... --warmUp=500 --duration=15 --iterations=10000`, GPU Compute Time mean. Pure CUDA kernel time, no host transfers.
- **Power:** `tegrastats --interval 1000`. "Silicon power" = `VDD_GPU_SOC` + `VDD_CPU_CV`. Carrier-board rails (`VIN_SYS_5V0`) excluded so the comparison is chip-vs-chip. Idle samples taken with no inference running, 20 s of 1 Hz tegrastats per mode.
- **Engine:** Path A FP16 built from the Ultralytics ONNX (Open Neural Network Exchange) export, `BuilderFlag.FP16`, no calibration. **Same engine binary in both modes**, only the runtime power cap differs.
- **Hardware:** AGX Orin Developer Kit reflashed as Nano 8 GB (`jetson-agx-orin-devkit-as-nano8gb internal`, NVIDIA-supported emulation), nvpmodel definitions match standard Nano 8 GB JetPack 6.2.2. **Caveat**: NVIDIA describes the emulation as performance-accurate; it does not model thermal throttling identically. Direct measurement on a retail Orin Nano dev kit would close this caveat.

## Conclusion

Power mode names describe ceilings, not workloads. A "7 W mode" deployment does not run at 7 W; it runs at whatever the workload draws under the cap. For YOLO26 at batch 1, both modes stay within a 7 W silicon envelope. 15 W mode draws 0.3-1.7 W more than 7 W mode does, but delivers 10-110 % more FPS and higher FPS per watt for every variant except `n`. 7 W mode is paying the throughput cost of enforcing a cap that this workload rarely approached.

Different workload, different result. The methodology generalises: before picking a power mode by which nameplate matches your power budget, measure your workload's draw in each mode. The lower cap may throttle you without delivering proportional power savings; the higher cap may not engage at all. The cost of measuring is one benchmark run per mode.

The companion case study is [YOLO26 at ~5 W: Hailo-8L, Hailo-8, and Jetson Orin Nano Compared](../yolo26-jetson-vs-hailo/), which uses the 7 W mode here to power-match against the Hailo platforms.
