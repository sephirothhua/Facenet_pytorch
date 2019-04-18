#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import os
from random import randint
import numpy as np
import cv2
try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

try:
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
# MAX_WORKSPACE = 1 << 30
# MAX_BATCHSIZE = 1



# DATA='/mnist/'
# MODEL='resnet50.engine'
# API CHANGE: Try to generalize into a utils function
#Run inference on device



def infer(engine, input_img, batch_size):
    #load engine
    context = engine.create_execution_context()
    assert(engine.get_nb_bindings() == 2)
    #create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    #Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype = np.float32)

    #alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    #transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    #execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    #return predictions
    return output


def infer_new(d_input,stream,context,bindings,output,d_output, input_img, batch_size):
    #transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    #execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    #return predictions
    return output

def get_testcase(path):
    im = Image.open(path)
    assert(im)
    arr = np.array(im)
    #make array 1D
    img = arr.ravel()
    return img

#Also prints case to console
def normalize(data):
    #allocate pagelocked memory
    norm_data = cuda.pagelocked_empty(data.shape, np.float32)
    print("\n\n\n---------------------------", "\n")
    for i in range(len(data)):
        print(" .:-=+*#%@"[data[i] // 26] + ("\n" if ((i + 1) % 28 == 0) else ""), end="");
        norm_data[i] = 1.0 - data[i] / 255.0
    print("\n")
    return norm_data


# def transfer(image):




class Engine_Config():
    batch_size = 1
    input_size = 3*56*56*4 # channel * width * height * sizeof(float32)
    width = 56
    height = 56
    output_size = 128*4 # features * sizeof(float32)
    engine_path = "./engine/face_engine.engine"


def main():

    # Get the engine configuration
    cfg = Engine_Config()

    # Load Engine
    engine = trt.utils.load_engine(G_LOGGER,cfg.engine_path)
    assert(engine),"No Engine loaded!"
    context = engine.create_execution_context()
    assert (engine.get_nb_bindings() == 2)
    # Create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * 1
    # Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)
    # Allocate device memory
    d_input = cuda.mem_alloc(cfg.batch_size * cfg.input_size)
    d_output = cuda.mem_alloc(cfg.batch_size * cfg.output_size)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    # Interface a picture
    image = cv2.imread('head_data/001/1/0_3.jpg')
    image = cv2.cvtColor(cv2.resize(image,(cfg.width,cfg.height)),cv2.COLOR_BGR2RGB) / 255.0
    image = np.reshape(image,(3,56,56))
    pre = infer_new(d_input, stream, context, bindings, output, d_output, image, cfg.batch_size)
    print(pre)



if __name__ == "__main__":
    main()