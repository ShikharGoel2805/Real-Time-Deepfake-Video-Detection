import timm
import torch.nn as nn

def build_model(backbone="tf_efficientnet_b0_ns", num_classes=1):
    m = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    # make sure we output a single logit for BCE
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Linear):
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        m.fc = nn.Linear(m.fc.in_features, 1)
    return m
