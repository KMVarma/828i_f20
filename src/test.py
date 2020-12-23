import tensorflow as tf
from .perturbation import adv_chroma_and_freq_gradient
from .util import rgb_to_ycc, ycc_to_rgb
from .dct import blockwise_dct, blockwise_idct

### MOBILENET SETUP ###

# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[tf.newaxis, ...]
    return image


# Input: an image of shape [1, H, W, 3] with RGB values in [-1, 1]
# Output: an image of shape [H, W, 3] with YCC values in [0, 1]
def unpreprocess(im):
    mod_im = im[0]
    mod_im = (mod_im + 1) / 2.0
    mod_im = rgb_to_ycc(mod_im)

    return mod_im


def get_doggo_pic():
    image_path = tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    )
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)

    return image


# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights="imagenet")
pretrained_model.trainable = False

# Classifies an image of dimension [H, W, 3] with RGB values in [0, 1]
def classify(im):
    # Mobilenet expects images of the form [1, H, W, 3] with RGB values in [-1, 1]
    mod_im = im[tf.newaxis, ...]
    mod_im = (mod_im - 0.5) * 2.0
    mod_im = tf.clip_by_value(mod_im, -1, 1)

    return pretrained_model(mod_im)


### TESTING SECTION ###

import matplotlib as mpl
import matplotlib.pyplot as plt

doggo = get_doggo_pic()
doggo_probs = pretrained_model.predict(doggo)
labrador_class = 208

doggo_ycc = unpreprocess(doggo)


def show_gradient(grad):
    shifted = grad * 0.5 + 0.5
    clipped = tf.clip_by_value(shifted, 0, 1)
    rgb = ycc_to_rgb(clipped)
    plt.imshow(rgb)


def run():
    # Do a DCT round trip and make sure the output still looks like a dog
    doggohat = blockwise_dct(doggo_ycc * 255 - 128, 28)
    doggoroundtrip = (blockwise_idct(doggohat, 28) + 128) / 255
    plt.figure()
    plt.imshow(ycc_to_rgb(doggoroundtrip))
    plt.title("doggo DCT roundtrip")
    plt.show()

    # Compute the gradients and plot them
    chroma_grad, freq_grad = adv_chroma_and_freq_gradient(doggo_ycc, labrador_class, classify, 28)

    plt.figure()
    show_gradient(chroma_grad)
    plt.title(f"Chroma gradient (l2 norm {tf.norm(chroma_grad)})")
    plt.show()
    plt.savefig("chroma_grad.png")

    plt.figure()
    show_gradient(freq_grad)
    plt.title(f"Freq gradient (l2 norm {tf.norm(freq_grad)})")
    plt.show()
    plt.savefig("freq_grad.png")

    print("Saved figures")


if __name__ == "__main__":
    run()
