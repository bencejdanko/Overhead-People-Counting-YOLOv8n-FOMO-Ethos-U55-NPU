import numpy as np
import tensorflow as tf

def convert_to_int8_tflite(model, calibration_images, save_path='fomo_production_int8.tflite'):
    def representative_dataset():
        for i in range(min(150, len(calibration_images))):
            data = np.expand_dims(calibration_images[i], axis=0).astype(np.float32)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Quantized architecture exported successfully to {save_path}")