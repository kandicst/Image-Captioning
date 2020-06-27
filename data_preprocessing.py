import tensorflow as tf
import os
import json

from layers.VGG19 import VGG19


def create_feature_vectors():
    vgg = VGG19()

    print(vgg.layers)
    pass


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
        # print('<START> ' + element['caption'] + ' <END>' + ' ' + str(element['image_id']))

    with open(out_file + '/captions.json', 'w') as f1:
        f1.write(json.dumps(captions))

    with open(out_file + '/image_ids.json', 'w') as f2:
        f2.write(json.dumps(image_ids))

    return captions, image_ids


if __name__ == "__main__":
    # process_train_test_captions()
    pass
