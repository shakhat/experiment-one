# YOLO inference with ONNX runtime

YOLO is a family of computer vision models provided by `ultralytics` project. 
This folder contains an example of how to run YOLO model inference in ONNX runtime without 
dependency on `torch`.

ONNX runtime is responsible for running model tensor operations only, while 
both pre- and post-processing stages are performed outside. 
However since all YOLO models have the same tensor interface it is possible to reuse the
same processing code for different models.

## Contents
 * `yolo_onnx_inference.py` - YOLO model inference with ONNX Runtime on CPU / CUDA / Ascend NPU

## How to run
 1. Export YOLO model to ONNX format: 
    * CPU & CUDA - `yolo export model=yolov8n.pt format=onnx dynamic=true`. 
    * Ascend NPU - `yolo export model=yolov8n.pt format=onnx dynamic=true opset=15`.

 2. Install dependencies:
    * CPU - `pip install -r requirements_cpu.txt`
    * CUDA - `pip install -r requirements_cuda.txt`
    * Ascend NPU - `pip install -r requirements_npu.txt`

 3. Run the inference: 
    * CPU - `python yolo_onnx_inference.py --model yolov8n.onnx --img bus.jpg --out output.jpg --device cpu`
    * CUDA - `python yolo_onnx_inference.py --model yolov8n.onnx --img bus.jpg --out output.jpg --device cuda`
    * Ascend NPU - `python yolo_onnx_inference.py --model yolov8n.onnx --img bus.jpg --out output.jpg --device npu`

ONNX runtime reports what provider is used to execute tensor operations. For example, 
if Ascend NPU is configured properly then the following line should appear in logs:
`All nodes placed on [CANNExecutionProvider]. Number of nodes: 1`.

When executed for the first time, Ascend NPU runtime recompiles a model into internal format.
This operation may take up to a few minutes, the internal model representation is cached in a local file 
with name `CANNExecutionProvider_main_graph_xxxx`, following runs use the cached model and are much faster.

## CLI

```
$ python yolo_onnx_inference.py --help
usage: yolo_onnx_inference.py [-h] [--device DEVICE] --model MODEL --img IMG --out OUT [--verbose] [--profiling]

options:
  -h, --help       show this help message and exit
  --device DEVICE  CPU or AI accelerator
  --model MODEL    Path to ONNX model
  --img IMG        Path to the input image
  --out OUT        Path to the output image
  --verbose        Enable verbose logging
  --profiling      Enable ONNX profiling
```
