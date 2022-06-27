from torchvision import models
import torch

model = models.convnext_tiny(pretrained=False, num_classes=2, activation='sigmoid')
pth_checkpoint = '/home/vid/hdd/projects/PycharmProjects/torchvision_classifier/ckp87.pth'
weights = torch.load(pth_checkpoint, map_location="cpu")
# model.load_state_dict(weights.state_dict())
model.load_state_dict(weights)
model.to("cpu")
model.eval()

x = torch.randn(1, 3, 112, 112, requires_grad=True)
torch.onnx.export(model, x, "87.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  )
