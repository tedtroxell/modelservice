from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone #, _validate_trainable_layers


class ImageDetector(FasterRCNN):
    def __init__(self):
        # a = _validate_trainable_layers(True, None, 5, 3)
        super(self.__class__, self).__init__(resnet_fpn_backbone('resnet50', True, trainable_layers=5),91)