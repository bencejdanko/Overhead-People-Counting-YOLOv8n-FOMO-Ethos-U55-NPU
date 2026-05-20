import tensorflow as tf
from tensorflow.keras import layers, models, applications

def production_weighted_bce_loss(pos_weight=24.0):
    def loss(y_true, y_pred_logits):
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred_logits,
            pos_weight=pos_weight
        )
    return loss

def build_and_compile_fomo(input_shape=(192, 192, 3), lr=0.0005, pos_weight=24.0):
    inputs = layers.Input(shape=input_shape, name="image_input")
    backbone = applications.MobileNetV2(
        input_tensor=inputs, alpha=0.35, include_top=False, weights='imagenet'
    )
    x = backbone.get_layer('block_6_expand_relu').output
    outputs = layers.Conv2D(filters=1, kernel_size=1, activation=None, name='fomo_head')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="FOMO_Production")
    
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(from_logits=True, name="loss_bce"),
        tf.keras.metrics.Precision(thresholds=0.0, name="precision"),
        tf.keras.metrics.Recall(thresholds=0.0, name="recall")
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=production_weighted_bce_loss(pos_weight=pos_weight),
        metrics=metrics
    )
    return model