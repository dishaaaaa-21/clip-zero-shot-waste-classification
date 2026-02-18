import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_linear_probe(model, preprocess, device, dataset):
    X = []
    y = []

    with torch.no_grad():
        for image, label in tqdm(dataset):
            image = preprocess(image).unsqueeze(0).to(device)

            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            X.append(emb.cpu().numpy())
            y.append(label)

    X = np.vstack(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    return acc
