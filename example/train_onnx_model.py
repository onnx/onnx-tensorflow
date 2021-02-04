import os
import onnx
import logging
import numpy as np

import tensorflow as tf
tf_compat = tf.compat.v1
from tensorflow.keras import datasets, layers, models
import onnx_tf

batch_size = 32
epochs = 4

saved_model_path = './saved_model/'
onnx_model_file = './onnx_model/model.onnx'
trained_onnx_model = './onnx_model/trained.onnx'
onnx_model_path = os.path.dirname(onnx_model_file)


def get_dataset():
  dataset = datasets.cifar10

  (x_train, y_train), (x_test, y_test) = dataset.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(batch_size, drop_remainder=True)
  test_ds = tf.data.Dataset.from_tensor_slices(
      (x_test, y_test)).batch(batch_size, drop_remainder=True)
  return train_ds, test_ds


def save_trained_onnx(tensor_dict, onnx_model, sess):
  # Collect retrained parameters.
  retrained_params = {}
  for name, tensor in tensor_dict.items():
    if isinstance(tensor, tf.Variable):
      retrained_params[name] = sess.run(tensor)

  # Update onnx model using new parameters:
  from onnx import mapping
  for tensor in onnx_model.graph.initializer:
    if tensor.name in retrained_params:
      print("Updating {}.".format(tensor.name))
      assert tensor.HasField("raw_data")
      tensor.raw_data = retrained_params[tensor.name].tobytes()

  onnx.save(onnx_model, trained_onnx_model)


def train_tf_model():
  model = models.Sequential()
  model.add(
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))

  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))

  if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
  model.save(saved_model_path)
  model.summary()
  print("tf saved model: {}".format(saved_model_path))
  '''
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    model.save(saved_model_path)
    '''


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
                loss, accuracy, _ = sess.run([loss_op, eval_op, opt_op],
                                             feed_dict=feed_dict)
                if (step % 100) == 0:
                  print('Epoch {}, test* step {}, loss:{}, accuracy:{}'.format(
                      epoch, step, loss, accuracy))
                step += 1
              except tf.errors.OutOfRangeError:
                break
            break
      save_trained_onnx(tf_rep.tensor_dict, onnx_model, sess)


def run_trained_onnx_model():
  onnx_model = onnx.load(trained_onnx_model)
  tf_rep = onnx_tf.backend.prepare(onnx_model, logging_level=logging.ERROR)
  input_name = tf_rep.inputs[0]
  train_data, test_data = get_dataset()

  for img, label in test_data:
    input = img.numpy().astype('float32')
    gt = label.numpy()
    break

  output = tf_rep.run({input_name: input})

  pred = np.argmax(output[0], axis=1).tolist()
  print("labels:{}".format(gt.flatten().tolist()))
  print("  pred:{}".format(pred))


if __name__ == "__main__":
  train_tf_model()
  convert_tf2onnx()
  train_onnx_model()
  run_trained_onnx_model()
