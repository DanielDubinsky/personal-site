# Porting YOLO26n to the Hailo-8L: Results & Chronology

## Summary & Performance Results

This project successfully ported the YOLO26n model to the Hailo-8L AI accelerator using a hybrid architecture(Hailo + CPU). Below are the key performance metrics achieved on a Raspberry Pi 5.

### Performance Metrics

The C++ implementation achieves a ~2.3x speedup over the Python baseline and a ~12x speedup over the ONNX baseline.

**Table 1: End-to-End Performance**

| Metric | ONNX (CPU) | Python (Hailo + CPU) | C++ (Hailo + CPU) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **mAP** | 0.382 | 0.315 | 0.302 | Minor accuracy drop |
| **End-to-End Latency**| 144.21ms | 27.61ms | **11.92ms** | **>12x Speedup vs CPU** |
| **Frames Per Second (FPS)** | 6.93 | 36.22 | **83.88** | `1000 / Latency` |

The full code for this project is available on GitHub: [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo)

It includes:

*   Code to export YOLO26n to HEF
*   C++ inference implementation
*   Python inference implementation


### Accuracy

The drop in accuracy is likely due to the sensitivity of the non-NMS heads to quantization noise. In this hybrid architecture, the backbone is quantized to 8-bit integers, while the head runs in floating point. Small quantization errors in the feature maps output by the backbone can propagate to the regression outputs in the head, which are highly sensitive in anchor-free detectors like this. However, more inspection is warranted, I would specifically start with the quantization ranges for the last layers.

### Latency Breakdown

**Table 2: Runtime Component Analysis**

| Component                | ONNX (CPU) | Python (Hailo + CPU) | C++ (Hailo + CPU) |
| :----------------------- | :------------------ | :------------------- | :---------------- |
| **Backbone**             | 127.75ms (88.5%)    | 11.42ms (41.4%)      | 11.19ms (93.9%)   |
| **Head + Post-proc**     | 8.29ms (5.7%)       | 8.29ms (30.0%)       | **0.73ms (6.1%)** |
| **Overhead**             | ~8.17ms (5.6%)      | ~7.90ms (28.6%)      | **~0.01ms (0%)**  |
| **Total Latency**        | **144.21ms**        | **27.61ms**          | **11.92ms**       |
| **Throughput (FPS)**     | **6.93 FPS**        | **36.22 FPS**        | **83.88 FPS**     |

The transition to C++ reduced the post-processing and bridge overhead from 15.8ms to 0.74ms. This 20x improvement is due to sparse operations like topK and indexing being much more efficient in C++ than in Python/onnx runtime.

### Theoretical Limits & Analysis

#### FLOPs & Theoretical Limits
To understand the efficiency of the port, it's useful to look at the theoretical compute floor.
Assuming the following:

1. Zero time spent moving data (memory bandwidth is infinite).
2. 100% utilization (every compute unit is working every clock cycle).
3. Zero software or driver overhead.


| Parameter | Value |
| --- | --- |
| **Theoretical TOPS (Hailo-8L)** | 13 TOPS |
| **YOLO26n Total FLOPs** | 5.4 Billion |
| **Theoretical Compute Floor** | ~0.41 ms |
| **Python Latency** | 27.61 ms |
| **C++ Latency** | 11.92 ms |

Additionally, the DFC automatically partitioned the backbone into **5 execution contexts** due to SRAM limits on the Hailo-8L. This adds fixed PCIe overhead to every frame as the control software switches contexts. According to Gemini, 5 execution contexts for a nano-sized model is too much, so a deeper dive is warranted.

---

## Introduction

This document outlines the chronological process of porting the Ultralytics YOLO26n model to the Hailo-8L AI accelerator. As this was, to my knowledge, the first attempt to run this model on the Hailo-8L, the process required significant debugging and a hybrid-architecture approach.

The goal is to present the stages in the order they were performed, detailing the challenges and successes at each step.

## Detailed Chronology

### 1. Convert YOLO26n to ONNX
The first step was to obtain a standard ONNX representation of the pre-trained YOLO26n model. This was achieved using the export functionality provided by Ultralytics.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", opset=11)
```

### 2. Attempt Initial Parsing
With the ONNX file, the next logical step was a direct conversion to the Hailo Archive (HAR) format using the Hailo Dataflow Compiler (DFC). This attempt failed.

The DFC reported that several operators within the model's detection head were not natively supported by the Hailo-8L NPU.

```bash
hailo_sdk_client.model_translator.exceptions.ParsingWithRecommendationException: Parsing failed. The errors found in the graph are:
UnsupportedOperationError in op /model.23/GatherElements: GatherElements operation is unsupported
UnsupportedOperationError in op /model.23/GatherElements_1: GatherElements operation is unsupported
UnsupportedReduceMaxLayerError in op /model.23/ReduceMax: Failed to create reduce max layer at vertex /model.23/ReduceMax. Reduce max is only supported on the features axis, and with keepdim=True
UnsupportedShuffleLayerError in op /model.23/Flatten: Failed to determine type of layer to create in node /model.23/Flatten
UnsupportedOperationError in op /model.23/TopK: TopK operation is unsupported
UnsupportedOperationError in op /model.23/TopK_1: TopK operation is unsupported
UnsupportedOperationError in op /model.23/Mod: Mod operation is unsupported
UnsupportedGatherLayerError in op /model.23/Gather_3: Can't find index
```

This failure confirmed that a direct, end-to-end deployment on the NPU was not possible.

### 3. Split the ONNX Graph
The solution was to partition the model. I manually split the ONNX graph into two distinct subgraphs:

1.  **Subgraph 1 (Backbone):** Contained the initial layers of the network up to the final feature maps. This section is computationally heavy and well-suited for the NPU.
2.  **Subgraph 2 (Head):** Contained the unsupported layers responsible for generating the final bounding box detections.

The intention was to run Subgraph 1 on the Hailo-8L and Subgraph 2 on the host CPU (Raspberry Pi 5).


### 4. Parse the Backbone Subgraph
The DFC was then used to parse only the first subgraph (the backbone). This attempt was successful, producing a parsed representation of the model.

### 5. Initial Quantization and End-to-End Test
Before proceeding with proper calibration, a "smoke test" was needed to validate the hybrid pipeline. I configured the DFC to perform a test quantization using **random weights**. This is not for accuracy, but purely to generate a deployable Hailo Executable Format (`.hef`) file. The goal was to confirm that the entire data flow—from input image to final tensor—was working correctly.

### 6. Deploy and Run the Hybrid Model
With the test `.hef` file, I assembled the full inference pipeline on the Raspberry Pi 5.

#### 6.1. Hybrid Execution
The pipeline was orchestrated in Python:

1.  **NPU Inference:** An input image is fed to the Hailo-8L, which runs the backbone model using **HailoRT**.
2.  **Bridge:** The tensor output from the Hailo-8L transposed using **NumPy**. This prepares the data for the second stage.
3.  **CPU Inference:** The processed tensor is fed into the **ONNX Runtime**, which executes the second subgraph (the head) on the RPi5's CPU to produce the final detections.

#### 6.2. Initial Runtime Measurement

A preliminary measurement of this non-optimized, randomly-quantized pipeline was taken to establish a baseline.

The initial, uncalibrated pipeline ran at approximately 27.61ms.

### 7. Calibrate with COCO Dataset
With the pipeline validated, the next step was to perform proper post-training quantization (PTQ) to achieve good accuracy.

*   **Dataset:** A subset of the COCO 2017 `train` split (1,024 images) was used for calibration.
*   **Strategy:** The DFC was configured for **Optimization Level 2**, which includes equalization and 4 epochs of fine-tuning.

#### 7.1. Resolve Environment Issues
Attempting to run the full quantization process revealed significant environment issues. The Hailo Docker container, running in a WSL2 environment, was not correctly using the host's NVIDIA RTX 4070 Ti for acceleration, causing the process to fall back to CPU and extending the estimated time from minutes to hours.

Two patches were required to fix this:

1.  **GPU Memory Check:** The DFC aborts if GPU VRAM usage is over 5%. This was patched to allow 50% usage, accommodating the Windows host's memory reservation.
    ```bash
    find $VIRTUAL_ENV -name "nvidia_smi_gpu_selector.py" | xargs sed -i 's/max_memory_utilization=0.05/max_memory_utilization=0.5/g'
    ```
2.  **WSL Driver Path:** The Docker container was not mounting the WSL-specific NVIDIA driver path. The container launch script was modified to include the `-v /usr/lib/wsl:/usr/lib/wsl` volume and `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib` environment variable.

3. **Hailo Docker Shell Script** The script to run the docker fails to see the NVIDIA GPU in wsl, adding the following line in the script solves this issue:

    ```bash
    # comment out old variable
    #readonly NVIDIA_GPU_EXIST=$(lspci | grep -i "\(vga\|3d\|display\|video\|graphics\).*nvidia")
    readonly NVIDIA_GPU_EXIST=$(nvidia-smi &> /dev/null && echo true || echo false)
    ```

### 8. Successful Calibration

After fixing the environment, the GPU-accelerated calibration process completed successfully, taking approximately 12 minutes. This produced a fully quantized and optimized `.hef` file for the model's backbone.

### 9. Single Image Test and Debugging
To verify the final `.hef` file, I ran a single test image through the full hybrid pipeline on the RPi5. The initial output was incorrect, which led to a debugging phase.

The primary issues discovered were:

*   **Tensor Shape Mismatch:** The output shape from the HailoRT NPU inference did not exactly match the input shape expected by the ONNX Runtime for the head model. This required careful reshaping with NumPy in the bridge step.
*   **Data Type Discrepancies:** The HailoRT NPU inference output was uint8, while the ONNX Runtime expected float32, however, Hailo API supports dequantization, so the code was modified to use it.

After debugging the bridge code, the pipeline produced valid bounding boxes on the test image.


### 10. The BGR vs RGB Bug
During the initial evaluation, I noticed the mAP was lower than expected. Upon inspection, I realized OpenCV reads images in BGR format by default, while the model was trained on and expects RGB images.

Fixing this simple channel swap yielded improvements for both the quantized hybrid model and the floating-point reference:

| Model | Metric | BGR Input (Bug) | RGB Input (Fix) |
| :--- | :--- | :--- | :--- |
| **Hybrid (Hailo)** | mAP | 0.306 | 0.315 |
| **Reference (ONNX)** | mAP | 0.373 | 0.382 |

### 11. Evaluation
The final step was to formally evaluate the performance and accuracy of the optimized hybrid model on the Raspberry Pi 5. This involved running the entire COCO `val` set through the pipeline. The detailed results are presented in the Summary at the beginning of this document.

### 12. Porting to C++
To eliminate the significant overhead of Python and the data bridge, I rewrote the inference pipeline in C++. This implementation uses C++ to interface directly with HailoRT, **eliminating the ONNX Runtime dependency entirely**.

I have implemented the postprocessing as a template with the following header:
```cpp
template <int... Is>
struct IntList {};

template <int... Strides, int... Grids>
std::vector<Detection> run_postprocess(
    IntList<Strides...>,
    IntList<Grids...>,
    const std::vector<const float*>& cls_tensors,
    const float* reg_tensor,
    float conf_threshold
)
```

This templated approach eases the porting process for other model variants (like YOLO26m or YOLO26l) in the future.

The results were a dramatic improvement in latency, jumping from ~38 FPS to over 85 FPS.

```text
[Results]
  Iterations: 1000
  Hailo Inference (Write+Read):
    Mean: 11.1868 ms
  CPU Post-processing:
    Mean: 0.73427 ms
  Total Latency:
    Mean: 11.9214 ms
  Estimated FPS (Serial): 83.8825
```

**Note on Accuracy:**
The C++ implementation showed a slight dip in mAP (0.302 vs 0.315).
```text
  AP             : 0.3024
  AP50           : 0.4801
  AP75           : 0.3263
```
My main suspicion is the image resizing implementation (e.g., interpolation flags).

### 13. Next Steps
To further optimize this pipeline, the following steps are planned:

1.  **Adaround Optimization:** Move to Optimization Level 4 to recover the ~6.7% AP loss observed in the hybrid model.
2.  **Investigate Accuracy Drop:** Determine why the C++ implementation is slightly less accurate than the Python version.
3.  **Context Reduction:** Experiment with DFC compiler settings to compress the model into fewer execution contexts.

## Conclusion

This iterative process of discovery, problem-solving, and validation demonstrates a complete, if preliminary, port of a next-generation object detection model to a resource-constrained edge device.