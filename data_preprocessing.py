import tensorflow as tf
import os

MSCOCO_IMG_URL = 'http://images.cocodataset.org/zips/train2014.zip'
MSCOCO_ANNOT_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'

def download_file(url, name=''):
    
    is_zip = url.split('.')[-1] == 'zip'
    print(is_zip)
    file = tf.keras.utils.get_file( fname=name,
                                    origin=url, 
                                    extract=is_zip)


if __name__ == "__main__":
    download_file(MSCOCO_ANNOT_URL, 'annotations.zip')
