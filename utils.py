import numpy as np
import os, urllib.request, tempfile
from PIL import Image
import json
import torch
import matplotlib.pyplot as plt


def load_image(path, size, processor=None):
    """ Loads, resizes and preprocesses image from disk """

    H, W = size
    img = Image.open(path)
    img = np.array(Image.fromarray(img).resize(H, W))

    if processor is not None:
        return processor(img)
    return img


def load_json(file):
    with open(file, 'r') as f1:
        return json.loads(f1.read())


def load_image_from_url(url):
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as file:
            file.write(f.read())

        img = Image.open(fname)
        os.remove(fname)
        return img
    except Exception as e:
        print("Error while loading image " + url)
        print(e.code)


def save_model(model, name='saved_models/model'):
    torch.save(model.state_dict(), name)


def load_model(model, name='saved_models/model'):
    model.load_state_dict(torch.load(name))

def plot_loss(loss_plot):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()