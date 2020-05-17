import numpy as np
import os, urllib.request, tempfile
from PIL import Image 

def load_image(path, size, processor=None):
    """ Loads, resizes and preprocesses image from disk """
    
    H, W = size
    img = Image.open(path)
    img = np.array(Image.fromarray(img).resize(H, W))

    if processor is not None:
        return processor(img)
    return img

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