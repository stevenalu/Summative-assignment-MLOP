import numpy as np
from preprocessing import load_image, get_class_names


def predict(model, file):
    x = load_image(file)
    probs = model.predict(x)[0]
    names = get_class_names()
    idx = int(np.argmax(probs))
    return {
        "label": names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {names[i]: float(probs[i]) for i in range(len(names))},
    }