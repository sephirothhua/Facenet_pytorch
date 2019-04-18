# FaceNet-Pytorch
This is a simpliest FaceNet trial using pytorch with Triplet Loss and Cross Entropy Loss.
 The backbone use the Resnet-19. It transforms a picture into 128 features and can be used in Reid or other project. 

The code shows how to convert a pytorch model to an engine for tensorrt to deploy.

Require the pytorch 1.0.1 version.
### 1.Directory Structure
```
project 
  |--README.md
  |--data_txt # Generate by the code
     |--train_data.txt
     |--train_label.txt
     |--val_data.txt
     |--val_label.txt 
  |--engine # The folder to save engine
     |--face_engine
  |--head_data # The folder for data  
     |--001 # The person number, just like 001,002
        |--001 # The folder to put person data(The folder name can be any and contain muti folders)
  |--log # The folder to save model
  |--logs # The folder to save tensorboard logs
  |--onnx_model # The folder to save onnx models
  |--test_model # The folder to save test models     
```

### 2. Train The Model
* Prepare the data

Prepare your data in head_data folder. The data must be ``head_data/PersonId/ImageFolders``. Then run
```angular2
python data2txt.py
```  
to generate the txt file in data_txt.

* Train the model

Change the ``Config.py`` and ``train.py`` for your requirements.
Then run
```
python train.py
```
for training.

### 3. Test The Model
Change the ``test.py`` for your requirement and then run ``python test.py`` for test.

### 4. Convert to ONNX model
Change the ``model_path`` in ``pytorch_to_onnx.py`` then run ``python pytorch_to_onnx.py`` for convert.

### 5. Convert ONNX model to Engine
It needs the [onnx-tensorrt package](https://github.com/onnx/onnx-tensorrt).

Then run
```
onnx2engine your_onnx_model.onnx -o your_engine_name.engine
```
for convert.

### 6. Test the engine file
Change the model dir in ``engine_test.py`` or ``engine_trt.py`` the run ``python engine_test.py`` or ``python engine_trt.py`` for engine test.