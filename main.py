from utils import load_clip
from zero_shot import run_zero_shot
from linear_probe import run_linear_probe
from torchvision.datasets import ImageFolder

def main():
    print("Loading CLIP...")
    model, preprocess, device = load_clip()

    print("Loading TrashNet from local folder...")
    dataset = ImageFolder(root="data/trashnet")

    print("Running Zero-Shot...")
    zs_acc = run_zero_shot(model, preprocess, device, dataset)
    print("Zero-Shot Accuracy:", zs_acc)

    print("Running Linear Probe...")
    lp_acc = run_linear_probe(model, preprocess, device, dataset)
    print("Linear Probe Accuracy:", lp_acc)

if __name__ == "__main__":
    main()
