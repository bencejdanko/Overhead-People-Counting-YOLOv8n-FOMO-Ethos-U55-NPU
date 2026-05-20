import tensorflow as tf
from tensorflow.keras import layers, models, applications

def fomo_weighted_categorical_crossentropy(object_weight=100.0):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon(),
        )
        class_weights = tf.constant([1.0, object_weight], dtype=y_pred.dtype)
        weights = tf.reduce_sum(y_true * class_weights, axis=-1)
        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(cross_entropy * weights)

    return loss


def object_precision(y_true, y_pred):
    y_true_obj = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_obj = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    true_positives = tf.reduce_sum(y_true_obj * y_pred_obj)
    predicted_positives = tf.reduce_sum(y_pred_obj)
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())


def object_recall(y_true, y_pred):
    y_true_obj = tf.cast(tf.argmax(y_true, axis=-1) == 1, tf.float32)
    y_pred_obj = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
    true_positives = tf.reduce_sum(y_true_obj * y_pred_obj)
    actual_positives = tf.reduce_sum(y_true_obj)
    return true_positives / (actual_positives + tf.keras.backend.epsilon())


def build_fomo_model(input_shape=(192, 192, 3), alpha=0.35, weights="imagenet", num_classes=2):
    inputs = layers.Input(shape=input_shape, name="image_input")
    backbone = applications.MobileNetV2(
        input_tensor=inputs,
        alpha=alpha,
        include_top=False,
        weights=weights,
    )
    x = backbone.get_layer('block_6_expand_relu').output
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation="softmax",
        name="fomo_head",
    )(x)
    return models.Model(inputs=inputs, outputs=outputs, name="FOMO_Production")


def build_and_compile_fomo(input_shape=(192, 192, 3), lr=0.001, object_weight=100.0):
    model = build_fomo_model(input_shape=input_shape)
    
    metrics = [
        object_precision,
        object_recall,
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=fomo_weighted_categorical_crossentropy(object_weight=object_weight),
        metrics=metrics
    )
    return model


def fomo_weighted_bce(pos_weight=100.0):
    return fomo_weighted_categorical_crossentropy(object_weight=pos_weight)


def production_weighted_bce_loss(pos_weight=100.0):
    return fomo_weighted_categorical_crossentropy(object_weight=pos_weight)
