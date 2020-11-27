import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import imshow

def plot_image_with_inference(image_path, preprocess_fn, predict_fn, get_imagenet_label_fn):
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = tf.image.adjust_brightness(image, delta=0)

    image = preprocess_fn(image)
    image_probs = predict_fn(image)

    plt.figure(figsize=(10,10))
    imshow(image)
    _, image_class, class_confidence = get_imagenet_label_fn(image_probs)
    plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
    plt.show()
    return image, image_probs
