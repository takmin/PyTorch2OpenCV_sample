# PyTorch Model to OpenCV Sample

OpenCV DNN module only support inference model but not training model.  If you want to use your own trained convolutional neural networks (CNN), you need to use deep learning frame works such as Tensorflow, Caffe, Torch, and to export it to OpenCV. 

Here are sample codes to train CNN model with PyTorch and to use it with OpenCV, following the steps below:
1. Train CNN model with PyTorch and save it. (train_LeNet.py)
2. Create CNN model for inference and load trained parameters. (save_LeNet_ONNX.py)
3. Save Inference model as ONNX format file. (save_LeNet_ONNX.py)
4. Load ONNX file from OpenCV. (opencv_LeNet.cpp)

These sample codes train LeNET5 with MNIST dataset.

If you want to train Keras+Tensorflow model and export it to OpenCV, see below:
https://github.com/takmin/Keras2OpenCV