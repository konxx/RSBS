from .lenet import LeNet5
from .simple_cnn import SimpleCNN


def build_model(name: str, num_classes: int):
    name = name.lower().strip()
    if name == "lenet":
        return LeNet5(num_classes=num_classes)
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {name}")
