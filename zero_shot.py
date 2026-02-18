import torch
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

PROMPT_TEMPLATES = [
    "a photo of {} waste",
    "a photo of discarded {}",
    "a close-up photo of {} trash",
    "a recyclable {} material",
    "an image of {} garbage"
]

def build_text_features(model, device):
    text_features = []

    with torch.no_grad():
        for c in CLASS_NAMES:
            prompts = [t.format(c) for t in PROMPT_TEMPLATES]
            tokens = clip.tokenize(prompts).to(device)

            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            class_emb = emb.mean(dim=0)
            class_emb = class_emb / class_emb.norm()

            text_features.append(class_emb)

    return torch.stack(text_features)

def run_zero_shot(model, preprocess, device, dataset):
    text_features = build_text_features(model, device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for image, label in tqdm(dataset):
            image = preprocess(image).unsqueeze(0).to(device)

            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ text_features.T).squeeze(0)
            pred = sims.argmax().item()

            y_true.append(label)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    return acc
