import json
from layers.ResNet import ResNet
import numpy as np
from PIL import Image
from torchvision import transforms
import urllib.request
import io

resnet = ResNet()


def preprocess_image(img_path):
    img = Image.open(img_path + '.jpg')
    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')
    img_vec = transform_image(img)
    return img_vec, img


def get_image_from_path(path):
    return Image.open(path + '.jpg')


def preprocess_image_web(img_link):
    fd = urllib.request.urlopen(img_link)
    image_file = io.BytesIO(fd.read())
    img = Image.open(image_file)
    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')
    img_vec = transform_image(img)
    return img_vec, img


def pycoco_preprocess(img, target):
    if len(np.array(img).shape) == 2:
        img = img.convert('RGB')
    img = transform_image(img)
    return img, target


def transform_image(img, input_size=256):
    img = transforms.Resize(input_size)(img)
    img = transforms.CenterCrop(input_size)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = resnet(img.unsqueeze(0))
    return img


def process_train_test_captions():
    save_captions('mscoco/train/annotations/captions_train2014.json', 'extracted_data/train')
    save_captions('mscoco/test/annotations/captions_val2014.json', 'extracted_data/test')


def save_captions(in_file, out_file):
    with open(in_file) as f:
        data = json.load(f)

    captions = []
    image_ids = []

    for element in data['annotations']:
        captions.append('<START> ' + element['caption'] + ' <END>')
        image_ids.append(element['image_id'])

    with open(out_file + '/captions.json', 'w') as f1:
        f1.write(json.dumps(captions))

    with open(out_file + '/image_ids.json', 'w') as f2:
        f2.write(json.dumps(image_ids))

    return captions, image_ids
