from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from AnchorUtils import anchor_grid
from DatasetMMP import MMP_Dataset
import numpy as np


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Define necessary components
    batch_size = 16
    grid = anchor_grid(10, 10, 32.0, [70, 100, 140, 200], [0.5, 1.0, 2.0])
    dataset = MMP_Dataset("dataset_mmp/train/",
                          batch_size=batch_size,
                          num_parallel_calls=6,
                          anchor_grid=grid,
                          threshhold=0.5)
    net = MobileNetV2(input_shape=(320, 320, 3), weights="imagenet", include_top=False)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Add another layer
    last_layer = net.layers[-1]
    new_layer = tf.keras.layers.Conv2D(filters=24,  # 24
                                       kernel_size=(1, 1),
                                       padding="same",
                                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005),
                                       name="Own_Layer")(last_layer.output)
    net = tf.keras.models.Model(net.input, new_layer)

    counter = 0
    output_shape = (batch_size, 10, 10, 4, 3)
    negative_ratio = tf.constant(10)

    for (filenames, imgs, label_grids, scores) in dataset():
        with tf.GradientTape() as tape:
            positive_values_count = tf.math.reduce_sum(label_grids)
            negative_samples = tf.random.uniform(output_shape, minval=0, maxval=1, dtype=tf.float32)

            negative_samples = tf.reshape(negative_samples, np.prod(list(output_shape)))
            negative_samples = tf.sort(negative_samples)
            negative_sample_count = positive_values_count * negative_ratio

            top_k = tf.math.top_k(negative_samples, k=negative_sample_count.numpy())
            negative_samples = negative_samples > top_k.values[-1].numpy()
            negative_samples = tf.cast(negative_samples, tf.float32)

            negative_samples = tf.reshape(negative_samples, output_shape)
            negative_samples = tf.add(negative_samples, tf.cast(label_grids, tf.float32))

            negative_samples = tf.clip_by_value(negative_samples, clip_value_min=-1000000, clip_value_max=1)
            res = net(imgs, training=False)
            res = tf.reshape(res, (batch_size, 10, 10, 4, 3, 2))  # ,2
            # res = res + tf.add_n(net.losses)
            # logits: 10x10x4x3x2  labels: 10x10x4x3
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label_grids, res)
            # 10 10 4 3
            loss = tf.math.multiply(loss, negative_samples)
            mean_loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(negative_samples)
            print(mean_loss.numpy(), counter)
        if counter == 2000:
            net.save('./models/mobilenet')
            return

        grads = tape.gradient(mean_loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        counter += 1


if __name__ == '__main__':
    main()
