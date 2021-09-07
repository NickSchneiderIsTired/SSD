from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from AnchorUtils import anchor_grid, hard_negative_samples
from DatasetMMP import MMP_Dataset
from constants import *
from test import main as eval


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Define necessary components
    batch_size = 16
    grid = anchor_grid(GRID_X, GRID_Y, GRID_SCALE, GRID_SIZES, GRID_RATIOS)
    dataset = MMP_Dataset("dataset_mmp/train/",
                          batch_size=batch_size,
                          num_parallel_calls=6,
                          anchor_grid=grid,
                          threshold=0.5)
    net = MobileNetV2(input_shape=(320, 320, 3), weights="imagenet", include_top=False)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Add another layer
    last_layer = net.layers[-1]
    new_layer = tf.keras.layers.Conv2D(filters=len(GRID_SIZES) * len(GRID_RATIOS) * 2,
                                       kernel_size=(1, 1),
                                       padding="same",
                                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005),
                                       name="Own_Layer")(last_layer.output)
    net = tf.keras.models.Model(net.input, new_layer)

    counter = 1
    output_shape = (batch_size, GRID_X, GRID_Y, len(GRID_SIZES), len(GRID_RATIOS))
    negative_ratio = tf.constant(10)

    for (filenames, imgs, label_grids, scores) in dataset():
        with tf.GradientTape() as tape:
            res = net(imgs, training=False)
            res = tf.reshape(res, output_shape + (2,))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label_grids, res)
            loss = loss + tf.add_n(net.losses)
            negative_samples = hard_negative_samples(loss, label_grids, output_shape, negative_ratio)
            loss = tf.math.multiply(loss, negative_samples)
            mean_loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(negative_samples)
            print(mean_loss.numpy(), counter)
        if counter == 500:
            net.save('./models/test')
            eval()
            return

        grads = tape.gradient(mean_loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        counter += 1


if __name__ == '__main__':
    main()
