from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from AnchorUtils import anchor_grid, create_label_grid, draw_rect
from DatasetMMP import MMP_Dataset
import numpy as np
from Evaluation import evaluate_net


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    # Define necessary components
    batch_size = 16
    grid = anchor_grid(10, 10, 32.0, [70, 100, 140, 200], [0.5, 1.0, 2.0])
    dataset = MMP_Dataset("dataset_mmp/train/",
                          batch_size=batch_size,
                          num_parallel_calls=4,
                          anchor_grid=grid,
                          threshhold=0.5)
    net = MobileNetV2(input_shape=(320, 320, 3), weights="imagenet", include_top=False)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # Add another layer
    last_layer = net.layers[-1]
    new_layer = tf.keras.layers.Conv2D(filters=24,
                                       kernel_size=(1, 1),
                                       padding="same",
                                       kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005),
                                       name="Own_Layer")(last_layer.output)
    net = tf.keras.models.Model(net.input, new_layer)
    # Make mobilenet layers untrainable

    counter = 0
    output_shape = (batch_size, 10, 10, 4, 3)
    negative_ratio = tf.constant(10)

    for (filenames, imgs, label_grids, scores) in dataset():
        with tf.GradientTape() as tape:
            positive_values_count = tf.math.reduce_sum(label_grids)
            negative_samples = tf.random.uniform(output_shape, minval=0, maxval=1, dtype=tf.float32)
            negative_sample_count = positive_values_count * negative_ratio
            top_k = tf.math.top_k(tf.sort(tf.reshape(negative_samples, np.prod(list(output_shape)))),
                                  k=negative_sample_count.numpy())
            negative_samples = negative_samples > top_k.values[-1].numpy()
            negative_samples = tf.cast(negative_samples, tf.float32)
            negative_samples = tf.add(negative_samples, tf.cast(label_grids, tf.float32))

            negative_samples = tf.clip_by_value(negative_samples, clip_value_min=-10000, clip_value_max=1)
            res = net(imgs, training=False)
            res = tf.reshape(res, (batch_size, 10, 10, 4, 3, 2))
            #regulization = tf.add_n(net.losses)
            # logits: 10x10x4x3x2  labels: 10x10x4x3
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label_grids, res)
            # 10 10 4 3

            loss = tf.math.multiply(loss, negative_samples)
            mean_loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(negative_samples)
            # print(counter)
            if counter % 10 == 0:
                print(mean_loss.numpy())
                evaluate_net(res, grid, imgs, counter)
                # test_net(res, grid, imgs[0], 'output' + str(counter) + '.jpg')
                #net.save('./models/mobilenet')

        grads = tape.gradient(mean_loss, net.trainable_weights)
        opt.apply_gradients(zip(grads, net.trainable_weights))
        counter += 1


def test_net(res, grid, img, out):
    numpy_img = img.numpy().astype('float32')
    res = res[:, :, :, :, :, 1]
    max = tf.nn.softmax(res[0]).numpy()
    label_grid = create_label_grid(max, 0.3)
    draw_rect(grid, tf.reshape(label_grid, (10, 10, 4, 3)).numpy(), numpy_img, out)


if __name__ == '__main__':
    main()
