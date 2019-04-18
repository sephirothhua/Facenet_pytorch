# model_path = "/data/github_test/binary_classifier/torch_logs_2/net_params000.pkl"
model_path = '/data/Project/Detect/Facenet_pytorch/test_model/checkpoint_epoch220.pth'

import torch
import cv2
from torch import nn
from ONNX_model import FaceNetModel
from torch.autograd import Variable


def torch_to_onnx(model, model_file):
    input_data = Variable(torch.randn(2, 3, 56, 56))
    # input_data1 = input_data[:, :, ::2, :, :]
    # input_data2 = input_data[:, :, ::16, :, :]
    torch.onnx.export(model, input_data, "{}.onnx".format(model_file), verbose=True)


#     torch.onnx._export(model, input_data,
#                        "{}.onnx".format(model_file),
#                        export_params=True,
#                        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
#                        verbose=True
#                       )

#     torch_out = torch.onnx._export(model,
#                             input,
#                             "model_caff2.onnx",
#                             export_params=False,
#                             operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
#                             )


model = FaceNetModel(embedding_size = 128)
model_dict = model.state_dict()
checkpoint = torch.load('./test_model/checkpoint_epoch220.pth',map_location='cuda:0')
checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
model_dict.update(checkpoint['state_dict'])
model.load_state_dict(model_dict)
model.eval()

if __name__ == "__main__":
    torch_to_onnx(model, model_path)
    # input_data = Variable(torch.randn(2, 3, 64, 112, 112))
    # output = model(input_data)

# import numpy as np
# np.expand_dims()
#
# import cv2
# import os
# cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# os.path.exists()