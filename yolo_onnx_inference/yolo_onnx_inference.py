# Ultralytics YOLO 🚀, AGPL-3.0 license
# Based on https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py

import argparse
import os
import time

import cv2
import numpy as np

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

import onnxruntime as ort  # this must go last because CANN version of onnxruntime interferes with torch


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_image, *, confidence_thres, iou_thres, verbose, profiling):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
            verbose: Enable verbose logging
            profiling: Enable ONNX profiling
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.verbose = verbose
        self.profiling = profiling

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x +
                                                     label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        def _round(x, n):
            return x - (x % n)
            
        def _transform_image(img):            
            # Get the height and width of the input image
            h = self.img_height = img.shape[0]
            w = self.img_width = img.shape[1]
            w = self.input_width = _round(w, 32)
            h = self.input_height = _round(h, 32)

            # Resize the image to match the input shape
            img = cv2.resize(img, (w, h))

            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0

            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
            
            image_data = image_data.astype(np.float32)
            return image_data
        
        ext = os.path.splitext(self.input_image)[1]
        if ext == '.mp4':
            cap = cv2.VideoCapture(self.input_image)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.img = frame  # store the last one
                frame = _transform_image(frame)
                frames.append(frame)
            image_data = np.stack(frames)
        else:
            # Read the input image using OpenCV
            self.img = cv2.imread(self.input_image)
            
            # Convert the image color space from BGR to RGB
            image_data = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            image_data = _transform_image(image_data)
            image_data = np.expand_dims(image_data, axis=0)
        
        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        output = output[-1:, :]  # get the last from the batch
        outputs = np.transpose(np.squeeze(output))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def inference(self, device):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        d_i = device.split(":")
        device_index = d_i[1] if len(d_i) > 1 else 0
        if device.startswith("npu"):
            providers = [
                ("CANNExecutionProvider", {"device_id": device_index,},),
                # For list of configuration options refer to 
                # https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#configuration-options
                # "device_id": 0, # The device ID, defaults to 0.
                # "npu_mem_limit": 2 * 1024 ** 3, # The size limit of the device memory arena in bytes.
                # "arena_extend_strategy": "kNextPowerOfTwo", # The strategy for extending the device memory arena.
                # "enable_cann_graph": False, # Whether to use the graph inference engine to speed up performance.
                # "dump_graphs": False, # Whether to dump the subgraph into onnx format for analysis of subgraph segmentation.
                # "dump_om_model": True, # Whether to dump the offline model for Ascend AI Processor to an .om file.
                # "precision_mode": "force_fp16", # The precision mode of the operator.
                # "op_select_impl_mode": "high_performance",  # Choose between high precision and high performance.
                # "optypelist_for_implmode": "Gelu", # Enumerate the list of operators which use the mode specified by the op_select_impl_mode parameter.
            ]
            device = f"cann:{device_index}" if len(d_i) > 1 else "cann"  # in ORT npu is called "cann"
        elif device.startswith("cuda"):
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_index,},),
                # For the list of configuration options refer to 
                # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
            ]
        else:
            providers = ["CPUExecutionProvider"]
            
        session_options = ort.SessionOptions()
        session_options.enable_profiling = self.profiling
        if self.verbose:
            session_options.log_severity_level = 0
            session_options.log_verbosity_level = 0
        
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, sess_options=session_options, providers=providers)

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Preprocess the image data
        img_data = self.preprocess()
                
        # Run inference using the preprocessed image data
        start = time.perf_counter_ns()
        outputs = session.run(None, {model_inputs[0].name: img_data})
        print(f'ONNX session.run is completed in {(time.perf_counter_ns() - start) / 10**6:.1f} ms')

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs[0])  # output image
    

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="CPU or AI accelerator")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--img", type=str, required=True, help="Path to the input image")
    parser.add_argument("--out", type=str, required=True, help="Path to the output image")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging")
    parser.add_argument("--profiling", action='store_true', help="Enable ONNX profiling")
    args = parser.parse_args()

    # Create an instance of the YOLOv8 class with the specified arguments
    yolo_model = YOLOv8(args.model, args.img, confidence_thres=0.5, iou_thres=0.5, verbose=args.verbose, profiling=args.profiling)

    # Perform object detection and obtain the output image
    output_image = yolo_model.inference(args.device)

    cv2.imwrite(args.out, output_image)
