from abc import ABC
import io
import base64
import torch
from PIL import Image
from captum.attr import IntegratedGradients
from .interface import HandlerInterface
from PIL import Image
import cv2
from skimage import io as skio

from torchvision import transforms

def url_to_image(url):
    image = skio.imread(url)
    return cv2.resize(image,(224,)*2)

class VisionHandler(HandlerInterface,ABC):
    """

    This is largely borrowed from the torchserve VisionHandler except it is extended to handle image urls as well

    :param HandlerInterface: [description]
    :type HandlerInterface: [type]
    :param ABC: [description]
    :type ABC: [type]
    :return: [description]
    :rtype: [type]
    """    
    image_processing = lambda x: x
    def initialize(self, context):
        super().initialize(context)
        self._initialize()
        self.ig = IntegratedGradients(self.model)
        self.initialized = True

    def _initialize(self):
        self.image_processing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image) if 'http' not in image else self.image_processing(url_to_image( image ))

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def get_insights(self, tensor_data, _, target=0):
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()