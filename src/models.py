import tensorflow as tf

def get_pretrained_model(model_name, weights):
    """
    Returns a pretrained model and a function to decode predictions of the model
    """

    if model_name == 'MobileNetV2':
        pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                             weights=weights)
        pretrained_model.trainable = False
        decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
    else:
        raise ValueError(f'{model_name} not yet supported')

    return pretrained_model, decode_predictions
