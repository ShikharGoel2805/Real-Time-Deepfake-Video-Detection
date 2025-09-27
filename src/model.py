# src/model.py
import timm
import torch.nn as nn

def build_model(
    backbone="tf_efficientnet_b0_ns",
    num_classes=2,           # <-- default to 2 for CrossEntropyLoss (real/fake)
    drop_rate=0.2,
    pretrained=True
):
    """
    Build a timm model with flexible backbone and output layer.

    Args:
        backbone (str): Name of the backbone model in timm.
        num_classes (int): Number of output classes (2 for binary classification with CrossEntropy).
        drop_rate (float): Dropout rate before classifier (if supported by backbone).
        pretrained (bool): Load pretrained weights.

    Returns:
        nn.Module: PyTorch model ready for training.
    """
    
    # timm expects num_classes matching final head size.
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    # Ensure head matches num_classes for common attribute names
    # (this is defensive: timm already sets num_classes head, but some custom models differ)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        if model.classifier.out_features != num_classes:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.out_features != num_classes:
            model.head = nn.Linear(model.head.in_features, num_classes)

    return model
