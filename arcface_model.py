"""
ArcFace Model Implementation for Face Recognition
Based on: https://github.com/spmallick/learnopencv/tree/master/Face-Recognition-with-ArcFace
"""

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import numpy as np


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        blocks = self.get_blocks(num_layers)

        if mode == 'ir':
            unit_module = BottleneckIR
        elif mode == 'ir_se':
            unit_module = BottleneckIRSE

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                          nn.Dropout(0.4),
                                          Flatten(),
                                          Linear(512 * 7 * 7, 512),
                                          BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                          nn.Dropout(0.4),
                                          Flatten(),
                                          Linear(512 * 14 * 14, 512),
                                          BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck[0], bottleneck[1], bottleneck[2]))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

    def get_blocks(self, num_layers):
        if num_layers == 50:
            blocks = [
                [(64, 64, 2), (64, 64, 1), (64, 64, 1)],
                [(64, 128, 2), (128, 128, 1), (128, 128, 1), (128, 128, 1)],
                [(128, 256, 2), (256, 256, 1), (256, 256, 1), (256, 256, 1), (256, 256, 1), (256, 256, 1)],
                [(256, 512, 2), (512, 512, 1), (512, 512, 1)]
            ]
        elif num_layers == 100:
            blocks = [
                [(64, 64, 2), (64, 64, 1), (64, 64, 1)],
                [(64, 128, 2), (128, 128, 1), (128, 128, 1), (128, 128, 1)],
                [(128, 256, 2)] + [(256, 256, 1)] * 13,
                [(256, 512, 2), (512, 512, 1), (512, 512, 1)]
            ]
        elif num_layers == 152:
            blocks = [
                [(64, 64, 2), (64, 64, 1), (64, 64, 1)],
                [(64, 128, 2), (128, 128, 1), (128, 128, 1), (128, 128, 1)],
                [(128, 256, 2)] + [(256, 256, 1)] * 23,
                [(256, 512, 2), (512, 512, 1), (512, 512, 1)]
            ]
        return blocks


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, 50, 'ir')
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, 50, 'ir_se')
    return model


def load_arcface_model(checkpoint_path='checkpoint/backbone.pth', device='cpu'):
    """
    Load pre-trained ArcFace model
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded ArcFace model
    """
    model = IR_SE_50(input_size=112)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✓ ArcFace model loaded from {checkpoint_path}")
        return model
    except FileNotFoundError:
        print(f"⚠ Model checkpoint not found at {checkpoint_path}")
        print("  Download from: https://drive.google.com/open?id=1H0ekPf3M3SzWivayY9oXQJ8GNcFn-O9q")
        return None
    except Exception as e:
        print(f"⚠ Error loading ArcFace model: {e}")
        return None


def extract_arcface_embedding(model, face_image, device='cpu'):
    """
    Extract 512-dimensional face embedding using ArcFace
    
    Args:
        model: Loaded ArcFace model
        face_image: Preprocessed face image (112x112)
        device: 'cpu' or 'cuda'
    
    Returns:
        embedding: 512-dimensional face embedding
    """
    if model is None:
        return None
    
    with torch.no_grad():
        # Ensure image is in correct format
        if isinstance(face_image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_image = face_image.astype(np.float32) / 255.0
            
            # Transpose to CxHxW
            face_image = np.transpose(face_image, (2, 0, 1))
            
            # Add batch dimension
            face_image = np.expand_dims(face_image, 0)
            
            # Convert to tensor
            face_tensor = torch.from_numpy(face_image).float()
        else:
            face_tensor = face_image
        
        face_tensor = face_tensor.to(device)
        embedding = model(face_tensor)
        embedding = embedding.cpu().numpy().flatten()
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return embedding


if __name__ == "__main__":
    # Test model loading
    import cv2
    
    print("Testing ArcFace model...")
    model = load_arcface_model()
    
    if model is not None:
        # Create dummy input
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = extract_arcface_embedding(model, dummy_face)
        
        if embedding is not None:
            print(f"✓ Successfully extracted embedding: {embedding.shape}")
            print(f"✓ ArcFace system is ready!")
        else:
            print("✗ Failed to extract embedding")
    else:
        print("✗ Model not loaded")
