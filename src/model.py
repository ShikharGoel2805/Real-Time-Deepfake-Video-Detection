import timm
import torch.nn as nn

def build_model(
    backbone="tf_efficientnet_b0_ns",
    num_classes=1,
    drop_rate=0.2,
    pretrained=True
):
    """
    Build a timm model with flexible backbone and output layer.

    Args:
        backbone (str): Name of the backbone model in timm.
        num_classes (int): Number of output classes (1 for binary, >1 for multi-class).
        drop_rate (float): Dropout rate before classifier (if supported by backbone).
        pretrained (bool): Load pretrained weights.

    Returns:
        nn.Module: PyTorch model ready for training.
    """
    
    # Determine output features
    out_features = 1 if num_classes == 1 else num_classes

    # Create timm model
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=out_features,
        drop_rate=drop_rate
    )

    # Replace final classifier for CNNs or ViTs
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, out_features)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, out_features)

    return model
