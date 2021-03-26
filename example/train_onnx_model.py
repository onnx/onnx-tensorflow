import os
import logging
import onnx
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import onnx_tf
tf_compat = tf.compat.v1

batch_size = 32
epochs = 2

saved_model_path = './saved_model/'
onnx_model_file = './onnx_model/model.onnx'
trained_onnx_model = './onnx_model/trained.onnx'
onnx_model_path = os.path.dirname(onnx_model_file)
use_dataset = 'mnist'  # mnist or cifar10
vgg_model = False


def get_dataset():
  if use_dataset == 'mnist':
    dataset = datasets.mnist
  else:
    dataset = datasets.cifar10

  (x_train, y_train), (x_test, y_test) = dataset.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  if use_dataset == 'mnist':
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(batch_size, drop_remainder=True)
  test_ds = tf.data.Dataset.from_tensor_slices(
      (x_test, y_test)).batch(batch_size, drop_remainder=True)
  return train_ds, test_ds


def save_trained_onnx(tensor_dict, onnx_model, sess):
  print('Update onnx model....')
  # Collect retrained parameters.
  retrained_params = {}
  for name, tensor in tensor_dict.items():
    if isinstance(tensor, tf.Variable):
      retrained_params[name] = sess.run(tensor)

  # Update onnx model using new parameters:
  for tensor in onnx_model.graph.initializer:
    if tensor.name in retrained_params:
      print("Updating {}.".format(tensor.name))
      assert tensor.HasField("raw_data")
      tensor.raw_data = retrained_params[tensor.name].tobytes()

  onnx.save(onnx_model, trained_onnx_model)
  print('Save trained onnx model {}'.format(trained_onnx_model))


class VGG16(models.Model):

  def __init__(self, input_shape):
    """
        :param input_shape: [32, 32, 3]
        """
    super(VGG16, self).__init__()

    weight_decay = 0.000
    self.num_classes = 10

    model = models.Sequential()

    model.add(
        layers.Conv2D(64, (3, 3),
                      padding='same',
                      input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(128, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(128, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(256, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(256, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(256, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(
        layers.Conv2D(512, (3, 3),
                      padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(
        layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(self.num_classes))
    # model.add(layers.Activation('softmax'))

    self.model = model

  def call(self, x):

    x = self.model(x)

    return x


def train_tf_model():
  if use_dataset == 'mnist':
    input_shape = (28, 28, 1)
    ds = datasets.mnist
  else:
    ds = datasets.cifar10
    input_shape = (32, 32, 3)

  if vgg_model:
    model = VGG16([32, 32, 3])
    model.build(input_shape=(None, 32, 32, 3))
  else:
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10))
    model.summary()

  (train_images, train_labels), (test_images, test_labels) = ds.load_data()
  train_images, test_images = train_images / 255.0, test_images / 255.0
  if use_dataset == 'mnist':
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

  model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  model.fit(train_images,
            train_labels,
            epochs=2,
            validation_data=(test_images, test_labels))
  model.evaluate(test_images, test_labels, verbose=2)
  if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
  model.save(saved_model_path)
  print("tf saved model: {}".format(saved_model_path))


def convert_tf2onnx():
  if not os.path.exists(onnx_model_path):
    os.mkdir(onnx_model_path)
  os.system('python -m tf2onnx.convert --saved-model {} --output {}'.format(
      saved_model_path, onnx_model_file))
  print('onnx model: {}'.format(onnx_model_file))


def train_onnx_model():
  onnx_model = onnx.load(onnx_model_file)
  tf_rep = onnx_tf.backend.prepare(onnx_model,
                                   training_mode=True,
                                   logging_level=logging.ERROR)
  training_flag_placeholder = tf_rep.tensor_dict[
      onnx_tf.backend.training_flag_name]
  input_name = onnx_model.graph.input[0].name
  output_name = onnx_model.graph.output[0].name

  with tf_rep.graph.as_default():
    with tf_compat.Session() as sess:
      y_truth = tf_compat.placeholder(tf.int64, [None], name='y-input')
      tf_rep.tensor_dict["y_truth"] = y_truth
      loss_op = tf.reduce_mean(
          tf_compat.losses.sparse_softmax_cross_entropy(
              labels=tf_rep.tensor_dict['y_truth'],
              logits=tf_rep.tensor_dict[output_name]))
      opt_op = tf_compat.train.AdamOptimizer().minimize(loss_op)
      eval_op = tf.reduce_mean(input_tensor=tf.cast(
          tf.equal(tf.argmax(input=tf_rep.tensor_dict[output_name], axis=1),
                   tf_rep.tensor_dict['y_truth']), tf.float32))

      train_data, test_data = get_dataset()
      sess.run(tf_compat.global_variables_initializer())
      print("==> Train the model..")

      for epoch in range(1, epochs + 1):
        step = 1
        next_batch = tf_compat.data.make_one_shot_iterator(
            train_data).get_next()
        while True:
          try:
            next_batch_value = sess.run(next_batch)
            feed_dict = {
                #tf_rep.tensor_dict[input_name]: next_batch_value[0].transpose((0, 3, 1, 2)),#for pytorch model
                tf_rep.tensor_dict[input_name]:
                    next_batch_value[0],
                tf_rep.tensor_dict['y_truth']:
                    next_batch_value[1].flatten()
            }
            feed_dict[training_flag_placeholder] = True
            loss, accuracy, _ = sess.run([loss_op, eval_op, opt_op],
                                         feed_dict=feed_dict)
            if (step % 100) == 0:
              print('Epoch {}, train step {}, loss:{}, accuracy:{}'.format(
                  epoch, step, loss, accuracy))
            step += 1
          except tf.errors.OutOfRangeError:
            step = 1
            next_batch = tf_compat.data.make_one_shot_iterator(
                test_data).get_next()
            while True:
              try:
                next_batch_value = sess.run(next_batch)
                feed_dict = {
                    #tf_rep.tensor_dict[input_name]: next_batch_value[0].transpose((0, 3, 1, 2)),#for pytorch model
                    tf_rep.tensor_dict[input_name]:
                        next_batch_value[0],
                    tf_rep.tensor_dict['y_truth']:
                        next_batch_value[1].flatten()
                }
                feed_dict[training_flag_placeholder] = False
                loss, accuracy = sess.run([loss_op, eval_op],
                                          feed_dict=feed_dict)
                if (step % 100) == 0:
                  print('Epoch {}, test* step {}, loss:{}, accuracy:{}'.format(
                      epoch, step, loss, accuracy))
                step += 1
              except tf.errors.OutOfRangeError:
                break
            break
      save_trained_onnx(tf_rep.tensor_dict, onnx_model, sess)


def run_onnx_model(onnx_file):
  print('Run onnx model....')
  onnx_model = onnx.load(onnx_file)
  tf_rep = onnx_tf.backend.prepare(onnx_model, logging_level=logging.ERROR)
  input_name = tf_rep.inputs[0]
  _, test_data = get_dataset()

  labels = []
  preds = []
  for img, label in test_data:
    input_value = img.numpy().astype('float32')
    gt = label.numpy()
    output = tf_rep.run({input_name: input_value})
    pred = np.argmax(output[0], axis=1).tolist()
    labels += gt.flatten().tolist()
    preds += pred

  correct_prediction = np.equal(preds, labels)
  acc = np.mean(correct_prediction)
  print('Accuracy: {}'.format(acc))


if __name__ == "__main__":
  train_tf_model()
  convert_tf2onnx()
  run_onnx_model(onnx_model_file)
  train_onnx_model()
  run_onnx_model(trained_onnx_model)
