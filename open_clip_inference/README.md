# Open CLIP model inference

Open CLIP is an open source implementation of OpenAI's CLIP (Contrastive Language-Image Pre-training) model. 
The model is trained to have both text and image embeddings in the same space, thus allowing to search 
for images by specifying a text description, or to search for similar images.

The following example script correlates image with text tokens and outputs probabilities of each individual token. 

## Contents
 * `open_clip_inference.py` - Open CLIP inference adapted for ONNX runtime on CPU / CUDA / Ascend NPU

## How to run
 1. Install dependencies:
    * CPU - `pip install -r requirements_cpu.txt`
    * CUDA - `pip install -r requirements_cuda.txt`
    * Ascend NPU - `pip install -r requirements_npu.txt`

 2. Export the model to ONNX format:
    `python export_to_onnx.py`
    The script creates 2 ONNX models, one for image and one for text transformer.

 3. Run the inference:
    * CPU - `python open_clip_inference.py --img bus.jpg --text "bus,man,cat,frog,turtle,car,street" --runtime onnx --device cpu`
    * CUDA - `python open_clip_inference.py --img bus.jpg --text "bus,man,cat,frog,turtle,car,street" --runtime onnx --device cuda`
    * Ascend NPU - `python open_clip_inference.py --img bus.jpg --text "bus,man,cat,frog,turtle,car,street" --runtime onnx --device npu:7` (device index 7)

### Run with "acl" runtime

The inference can be run in a native ACL mode. That requires a model in ".om" format. The model can be converted from ONNX format by "atc" tool, e.g.
`atc --model=open_clip_image_transformer.onnx --framework=5 --output=open_clip_image_transformer --input_shape="image:1,3,224,224" --soc_version=Ascend910ProB --mode=0` or can be created by invocation on onnx runtime. In the latter case ACL runtime creates 2 files with names starting with CANNExecutionProvider_main_graph_, 
one file is an image transformer model, the other is a text transformer. 


## CLI
```
$ python export_to_onnx.py --help
usage: export_to_onnx.py [-h] [--exporter {dynamo,legacy}] [--backport]

options:
  -h, --help            show this help message and exit
  --exporter {dynamo,legacy}
                        What exporter to use.
  --backport            Backport model to opset 15 (only used for dynamo exporter).
```

```
$ python open_clip_inference.py --help
usage: open_clip_inference.py [-h] --img IMG --text TEXT [--device DEVICE] [--runtime {onnx,pytorch,acl}] [--verbose]
                              [--profiling]

options:
  -h, --help            show this help message and exit
  --img IMG             Path to input image.
  --text TEXT           Comma-separated strings.
  --device DEVICE       Device to use for tensor operations.
  --runtime {onnx,pytorch,acl}
                        What runtime to use: ONNX, pytorch or native ACL.
  --verbose             Enable verbose logging.
  --profiling           Enable ONNX profiling
```
