# Porting YOLO26n to the Hailo-8L: Results & Chronology

## Summary & Performance Results

This project successfully ported the YOLO26n model to the Hailo-8L AI accelerator using a hybrid architecture. Below are the key performance metrics achieved on a Raspberry Pi 5.

### Performance Metrics

| Metric | Value | Notes |
| --- | --- | --- |
| **mAP (Before Quantization)** | 0.373 | On RPi5 CPU (ONNX Runtime) |
| **mAP (After Quantization)** | 0.306 | On RPi5 + Hailo-8L (Hybrid) |
| **End-to-End Latency**| 26.68ms | Avg. over 1000 iterations |
| **Frames Per Second (FPS)** | 38.09 | `1000 / Latency` |

The drop in accuracy is likely due to the sensitivity of the non-NMS heads to quantization noise. In this hybrid architecture, the backbone is quantized to 8-bit integers, while the head runs in floating point. Small quantization errors in the feature maps output by the backbone can propagate to the regression outputs in the head, which are highly sensitive in anchor-free detectors like this. However, more inspection is warranted, I would specifically start with the quantization ranges for the last layers.

### Latency Breakdown

| Component                | Hybrid (Hailo + CPU) | ONNX Only (CPU)     |
| ------------------------ | -------------------- | ------------------- |
| **Backbone**             | 11.93ms (44.7%)      | 138.75ms (86.1%)    |
| **Head**                 | 7.83ms (29.4%)       | 9.99ms (6.2%)       |
| **Overhead (Bridge/Copy)** | ~6.92ms (25.9%)      | ~12.47ms (7.7%)   |
| **Total Latency**        | **26.68ms**          | **161.21ms**        |
| **Throughput (FPS)**     | **38.09 FPS**        | **6.35 FPS**        |

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
| **Current Total Latency** | 26.68 ms |

#### The Hybrid Overhead
The "Bridge" (de-quantization, NumPy transpositions, and PCIe data movement) consumes nearly 26% of the total runtime. While the NPU backbone is extremely efficient (11.93 ms), the serial nature of moving data back to the CPU for the ONNX head creates a significant bottleneck.

Additionally, the DFC automatically partitioned the backbone into **5 execution contexts** due to SRAM limits on the Hailo-8L. This adds fixed PCIe overhead to every frame as the control software switches contexts.

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
2.  **Bridge:** The tensor output from the Hailo-8L is de-quantized and transposed using **NumPy**. This prepares the data for the second stage.
3.  **CPU Inference:** The processed tensor is fed into the **ONNX Runtime**, which executes the second subgraph (the head) on the RPi5's CPU to produce the final detections.

#### 6.2. Initial Runtime Measurement
A preliminary measurement of this non-optimized, randomly-quantized pipeline was taken to establish a baseline.

The initial, uncalibrated pipeline ran at approximately 11.93ms.

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
*   **Data Type Discrepancies:** The de-quantization process had to be precise to ensure the data type (e.g., `float32`) was correct for the ONNX Runtime.

After debugging the bridge code, the pipeline produced valid bounding boxes on the test image.

### 10. Evaluation
The final step was to formally evaluate the performance and accuracy of the optimized hybrid model on the Raspberry Pi 5. This involved running the entire COCO `val` set through the pipeline. The detailed results are presented in the Executive Summary at the beginning of this document.

### 11. Next Steps
To further optimize this pipeline, the following steps are planned:
1.  **Adaround Optimization:** Move to Optimization Level 4 to recover the ~6.7% AP loss observed in the hybrid model.
2.  **C++ Implementation:** Eliminate the ~6.92ms Python bridge overhead by moving to a native HailoRT/ORT C++ pipeline.
3.  **Context Reduction:** Experiment with DFC compiler settings to compress the model into fewer execution contexts.

## Conclusion
This iterative process of discovery, problem-solving, and validation demonstrates a complete, if preliminary, port of a next-generation object detection model to a resource-constrained edge device.