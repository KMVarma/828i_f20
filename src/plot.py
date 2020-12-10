import tensorflow as tf
import matplotlib.pyplot as plt

# Input:
#   * Path to image file
#   * A prediction function which takes in an RGB image with values in [0, 255] and returns the
#     predicted class label (a str value), and the confidence of the prediction (a float value)
def plot_image_with_inference(image_path, predict_fn):
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    label, confidence = predict_fn(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"{label} : {confidence*100:.02f}% Confidence")
    plt.show()
