# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Tool to inspect a model."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import os
import time

import torch
from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Tuple, List

from efficientdet1 import hparams_config
from efficientdet1 import inference
from efficientdet1 import utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import
import pickle
import requests
import io
import glob
from torchvision import transforms

def add_to_dict(d, k, v):
    if k in d.keys():
        d[k].append(v)
    elif k not in d.keys():
        d[k] = [v]
        # d[k].append(v)

    return d


class ModelInspector(object):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               export_ckpt: Text = None,
               saved_model_dir: Text = None,
               tflite_path: Text = None,
               batch_size: int = 1,
               hparams: Text = ''):
    self.model_name = model_name
    self.logdir = logdir
    self.tensorrt = tensorrt
    self.use_xla = use_xla
    self.ckpt_path = ckpt_path
    self.export_ckpt = export_ckpt
    self.saved_model_dir = saved_model_dir
    self.tflite_path = tflite_path

    model_config = hparams_config.get_detection_config(model_name)
    model_config.override(hparams)  # Add custom overrides
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    # If batch size is 0, then build a graph with dynamic batch size.
    self.batch_size = batch_size or None
    self.labels_shape = [batch_size, model_config.num_classes]

    height, width = model_config.image_size
    if model_config.data_format == 'channels_first':
      self.inputs_shape = [batch_size, 3, height, width]
    else:
      self.inputs_shape = [batch_size, height, width, 3]

    self.model_config = model_config

  def build_model(self,
                  inputs: tf.Tensor,
                  is_training: bool = False) -> List[tf.Tensor]:
    """Build model with inputs and labels and print out model stats."""
    logging.info('start building model')
    cls_outputs, box_outputs = inference.build_model(
        self.model_name,
        inputs,
        is_training_bn=is_training,
        config=self.model_config)

    # Write to tfevent for tensorboard.
    # train_writer = tf.summary.FileWriter(self.logdir)
    # train_writer.add_graph(tf.get_default_graph())
    # train_writer.flush()

    # all_outputs = list(cls_outputs.values()) + list(box_outputs.values())
    # return all_outputs

  def export_saved_model(self, **kwargs):
    """Export a saved model for inference."""
    tf.enable_resource_variables()
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.build()
    driver.export(self.saved_model_dir, tflite_path=self.tflite_path)


  def saved_model_inference_for_transformer(self, imgs, driver, layers, tipo='train', **kwargs):

      # preprocessing = transforms.ToPILImage()
      postprocessing = transforms.ToTensor()
      get_features = True
      output = {}
      output_detections = []
      # feats = []
      # raw_images = []
      for n, img in enumerate(imgs):
          # x = preprocessing(img.squeeze().cpu())
          x = img.squeeze().cpu().numpy().transpose(1, 2, 0)
          # x = preprocessing(img.cpu())
          # raw_images.append(np.array(x, dtype='uint8'))
          # Image.fromarray(x).convert("RGB").save("art1.png")
          # x.save("art1.png")
          # x = [np.array(x, dtype='uint8')]
          x = [x.astype(np.uint8)]
          # Image.fromarray(x[0]).convert("RGB").save("art1.png")
          detections_feat, detections_bs = driver.serve_images(x, get_features)#[2:]
          # x= ty
          # print(len(detections_feat))
          # print(detections_bs[0][:,0])
          # print(imgs.shape)
          ## det structure = [[image_ids, xmin, ymin, xmax, ymax, nmsed_scores, classes]]
          detections_bs[0][:, 0] = n


          # output_detections = np.concatenate(output_detections, detections_bs)
          output_detections.append(detections_bs[0])


          detections_feat.reverse()
          for k, i in zip(layers, detections_feat):
              v = postprocessing(i[0])
              output = add_to_dict(output, k, v)
              # print(k, len(output[k]), v.shape)

      # print(np.concatenate(output_detections, axis=0).shape)
      # quit()

          # print(i, output[i].shape)
      # print(len(feats))


      return output, np.concatenate(output_detections, axis=0)
  def saved_model_inference(self, dataset, driver, imgs, type='train', **kwargs):
    """Perform inference for the given saved model."""
    # driver = inference.ServingDriver(
    #     self.model_name,
    #     self.ckpt_path,
    #     batch_size=self.batch_size,
    #     use_xla=self.use_xla,
    #     model_params=self.model_config.as_dict(),
    #     **kwargs)
    # driver.load(self.saved_model_dir)

    # driver = inference.ServingDriver(
    #     inspector.model_name,
    #     inspector.ckpt_path,
    #     batch_size=inspector.batch_size,
    #     use_xla=inspector.use_xla,
    #     model_params=inspector.model_config.as_dict())
    #
    # driver.load(inspector.saved_model_dir)

    # Serving time batch size should be fixed.
    batch_size = self.batch_size or 1
    # all_files = list(tf.io.gfile.glob(image_path_pattern))
    feats = []
    dets = []
    ############################################################################################################
    if dataset == 'voc' or dataset == 'penn':

      all_files = imgs

      get_features=True
      num_batches = (len(all_files) + batch_size - 1) // batch_size

      for i in range(num_batches):
        batch_files = all_files[i * batch_size:(i + 1) * batch_size]
        height, width = self.model_config.image_size
        images = [Image.open(f).convert('RGB') for f in batch_files]
        if len(set([m.size for m in images])) > 0:
          # Resize only if images in the same batch have different sizes.
          images = [m.resize((height, width)) for m in images]
        raw_images = [np.array(m) for m in images]
        size_before_pad = len(raw_images)
        if size_before_pad < batch_size:
          padding_size = batch_size - size_before_pad
          raw_images += [np.zeros_like(raw_images[0])] * padding_size

        detections_bs = driver.serve_images(raw_images, get_features)
        feats.append(detections_bs[0])
        detections_bs[1][:, :, 0] = np.full(detections_bs[1].shape[:2], i)
        dets.append(np.reshape(detections_bs[1], (-1, 7)))

    elif dataset== 'coco' or dataset == 'lvis':
      all_files = imgs

      get_features = True
      # print('all_files=', all_files)
      num_batches = (len(all_files) + batch_size - 1) // batch_size

      for i in range(num_batches):
          # batch_files = all_files[i * batch_size:(i + 1) * batch_size]
          height, width = self.model_config.image_size
          # req = requests.get(data.loadImgs(ids=[imgs[i]])[0]['coco_url'])
          # images = [Image.open(f) for f in batch_files]
          # images = [Image.open(io.BytesIO(req.content)).convert('RGB')]
          if type=='val':
            for name in glob.glob('../datasets/coco/*/{:0>12d}.jpg'.format(imgs[i])):
              images = [Image.open(name).convert('RGB')]
          elif type == 'vis':
            images = [Image.open(imgs[i]).convert('RGB')]
          else:
            images = [Image.open('../datasets/coco/{}2017/{:0>12d}.jpg'.format(type,imgs[i])).convert('RGB')]
          rate = images[0].size[1]/images[0].size[0]
          if len(set([m.size for m in images])) > 1:
              # Resize only if images in the same batch have different sizes.
              # dim = max(height, images[0].size[1], images[0].size[0])
              # images = [m.resize((dim,dim)) for m in images]
              images = [m.resize((height, width)) for m in images]
            # images = [m.resize(( int(np.round(3*m.size[1]/2)), m.size[1])) for m in images]
          raw_images = [np.array(m) for m in images]
          size_before_pad = len(raw_images)
          if size_before_pad < batch_size:
              padding_size = batch_size - size_before_pad
              raw_images += [np.zeros_like(raw_images[0])] * padding_size

          detections_bs = driver.serve_images(raw_images, get_features)
          feats.append(detections_bs[0])
          detections_bs[1][:, :, 0] = np.full(detections_bs[1].shape[:2], i)
          dets.append(np.reshape(detections_bs[1], (-1,7)))


    else:
        all_files = list(tf.io.gfile.glob(imgs))
        get_features = False
    #########################################################################################################################

        # print('all_files=', all_files)
        num_batches = (len(all_files) + batch_size - 1) // batch_size

        for i in range(num_batches):
          batch_files = all_files[i * batch_size:(i + 1) * batch_size]
          height, width = self.model_config.image_size
          images = [Image.open(f) for f in batch_files]
          if len(set([m.size for m in images])) > 1:
            # Resize only if images in the same batch have different sizes.
            images = [m.resize((height, width)) for m in images]
          raw_images = [np.array(m) for m in images]
          size_before_pad = len(raw_images)
          if size_before_pad < batch_size:
            padding_size = batch_size - size_before_pad
            raw_images += [np.zeros_like(raw_images[0])] * padding_size

      ################################################## Mi código #####################################

          detections_bs = driver.serve_images(raw_images, get_features)

          feats.append(detections_bs[0])
    # print(all_files[i * batch_size:(i + 1) * batch_size])
    return feats, np.concatenate(dets)
  ###################################################################################

  def saved_model_benchmark(self,
                            image_path_pattern,
                            trace_filename=None,
                            **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict(),
        **kwargs)
    driver.load(self.saved_model_dir)
    raw_images = []
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    if len(all_files) < self.batch_size:
      all_files = all_files * (self.batch_size // len(all_files) + 1)
    raw_images = [np.array(Image.open(f)) for f in all_files[:self.batch_size]]
    driver.benchmark(raw_images, trace_filename)

  def saved_model_video(self, video_path: Text, output_video: Text, **kwargs):
    """Perform video inference for the given saved model."""
    import cv2  # pylint: disable=g-import-not-at-top

    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=1,
        use_xla=self.use_xla,
        model_params=self.model_config.as_dict())
    driver.load(self.saved_model_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(video_path))

    out_ptr = None
    if output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(output_video,
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = [np.array(frame)]
      detections_bs = driver.serve_images(raw_frames)
      new_frame = driver.visualize(raw_frames[0], detections_bs[0], **kwargs)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  def inference_single_image(self, image_image_path, output_dir, **kwargs):
    driver = inference.InferenceDriver(self.model_name, self.ckpt_path,
                                       self.model_config.as_dict())
    driver.inference(image_image_path, output_dir, **kwargs)

  def build_and_save_model(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      # Run the model
      inputs_val = np.random.rand(*self.inputs_shape).astype(float)
      labels_val = np.zeros(self.labels_shape).astype(np.int64)
      labels_val[:, 0] = 1

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        sess.run(tf.global_variables_initializer())
        # Run a single train step.
        sess.run(outputs, feed_dict={inputs: inputs_val})

      all_saver = tf.train.Saver(save_relative_paths=True)
      all_saver.save(sess, os.path.join(self.logdir, self.model_name))

      tf_graph = os.path.join(self.logdir, self.model_name + '_train.pb')
      with tf.io.gfile.GFile(tf_graph, 'wb') as f:
        f.write(sess.graph_def.SerializeToString())

  def eval_ckpt(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      self.build_model(inputs, is_training=False)
      inference.restore_ckpt(sess, self.ckpt_path,
                             self.model_config.moving_average_decay,
                             self.export_ckpt)

  def freeze_model(self) -> Tuple[Text, Text]:
    """Freeze model and convert them into tflite and tf graph."""
    with tf.Graph().as_default(), tf.Session() as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      if self.ckpt_path:
        # Load the true weights if available.
        inference.restore_ckpt(sess, self.ckpt_path,
                               self.model_config.moving_average_decay,
                               self.export_ckpt)
      else:
        # Load random weights if not checkpoint is not available.
        self.build_and_save_model()
        checkpoint = tf.train.latest_checkpoint(self.logdir)
        logging.info('Loading checkpoint: %s', checkpoint)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

      output_node_names = [node.op.name for node in outputs]
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names)

      tf_graph = os.path.join(self.logdir, self.model_name + '_frozen.pb')
      tf.io.gfile.GFile(tf_graph, 'wb').write(graphdef.SerializeToString())

    return graphdef

  def benchmark_model(self,
                      warmup_runs,
                      bm_runs,
                      num_threads,
                      trace_filename=None):
    """Benchmark model."""
    if self.tensorrt:
      print('Using tensorrt ', self.tensorrt)
      graphdef = self.freeze_model()

    if num_threads > 0:
      print('num_threads for benchmarking: {}'.format(num_threads))
      sess_config = tf.ConfigProto(
          intra_op_parallelism_threads=num_threads,
          inter_op_parallelism_threads=1)
    else:
      sess_config = tf.ConfigProto()

    # rewriter_config_pb2.RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.dependency_optimization = 2
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      output = self.build_model(inputs, is_training=False)

      img = np.random.uniform(size=self.inputs_shape)

      sess.run(tf.global_variables_initializer())
      if self.tensorrt:
        fetches = [inputs.name] + [i.name for i in output]
        goutput = self.convert_tr(graphdef, fetches)
        inputs, output = goutput[0], goutput[1:]

      if not self.use_xla:
        # Don't use tf.group because XLA removes the whole graph for tf.group.
        output = tf.group(*output)
      else:
        output = tf.add_n([tf.reduce_sum(x) for x in output])

      output_name = [output.name]
      input_name = inputs.name
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_name)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      tf.import_graph_def(graphdef, name='')

      for i in range(warmup_runs):
        start_time = time.time()
        sess.run(output_name, feed_dict={input_name: img})
        logging.info('Warm up: {} {:.4f}s'.format(i, time.time() - start_time))

      print('Start benchmark runs total={}'.format(bm_runs))
      start = time.perf_counter()
      for i in range(bm_runs):
        sess.run(output_name, feed_dict={input_name: img})
      end = time.perf_counter()
      inference_time = (end - start) / 10
      print('Per batch inference time: ', inference_time)
      print('FPS: ', self.batch_size / inference_time)

      if trace_filename:
        run_options = tf.RunOptions()
        run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()
        sess.run(
            output_name,
            feed_dict={input_name: img},
            options=run_options,
            run_metadata=run_metadata)
        logging.info('Dumping trace to %s', trace_filename)
        trace_dir = os.path.dirname(trace_filename)
        if not tf.io.gfile.exists(trace_dir):
          tf.io.gfile.makedirs(trace_dir)
        with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

  def convert_tr(self, graph_def, fetches):
    """Convert to TensorRT."""
    from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    converter = trt.TrtGraphConverter(
        nodes_blacklist=[t.split(':')[0] for t in fetches],
        input_graph_def=graph_def,
        precision_mode=self.tensorrt)
    infer_graph = converter.convert()
    goutput = tf.import_graph_def(infer_graph, return_elements=fetches)
    return goutput

  def run_model(self, runmode, **kwargs):
    """Run the model on devices."""
    if runmode == 'dry':
      self.build_and_save_model()
    elif runmode == 'freeze':
      self.freeze_model()
    elif runmode == 'ckpt':
      self.eval_ckpt()
    elif runmode == 'saved_model_benchmark':
      self.saved_model_benchmark(
          kwargs['input_image'],
          trace_filename=kwargs.get('trace_filename', None))
    elif runmode in ('infer', 'saved_model', 'saved_model_infer',
                     'saved_model_video'):
      config_dict = {}
      if kwargs.get('line_thickness', None):
        config_dict['line_thickness'] = kwargs.get('line_thickness')
      if kwargs.get('max_boxes_to_draw', None):
        config_dict['max_boxes_to_draw'] = kwargs.get('max_boxes_to_draw')
      if kwargs.get('min_score_thresh', None):
        config_dict['min_score_thresh'] = kwargs.get('min_score_thresh')

      if runmode == 'saved_model':
        self.export_saved_model(**config_dict)
      elif runmode == 'infer':
        self.inference_single_image(kwargs['input_image'],
                                    kwargs['output_image_dir'], **config_dict)
      elif runmode == 'saved_model_infer':
        self.saved_model_inference(kwargs['input_image'],
                                   kwargs['output_image_dir'], kwargs['imagenes'], **config_dict)
      elif runmode == 'saved_model_video':
        self.saved_model_video(kwargs['input_video'], kwargs['output_video'],
                               **config_dict)
    elif runmode == 'bm':
      self.benchmark_model(
          warmup_runs=5,
          bm_runs=kwargs.get('bm_runs', 10),
          num_threads=kwargs.get('threads', 0),
          trace_filename=kwargs.get('trace_filename', None))
    else:
      raise ValueError('Unkown runmode {}'.format(runmode))


def main(args, img_list):
  # if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
  #   logging.info('Deleting log dir ...')
  #   tf.io.gfile.rmtree(FLAGS.logdir)

  inspector = ModelInspector(
      model_name=args.model_name,
      logdir=args.logdir,
      tensorrt=args.tensorrt,
      use_xla=args.xla,
      ckpt_path=args.ckpt_path,
      export_ckpt=args.export_ckpt,
      saved_model_dir=args.saved_model_dir,
      tflite_path=args.tflite_path,
      batch_size=args.batch_size,
      hparams=args.hparams)
  inspector.run_model(
      args.runmode,
      input_image=args.input_image,
      output_image_dir=args.output_image_dir,
      input_video=args.input_video,
      output_video=args.output_video,
      line_thickness=args.line_thickness,
      max_boxes_to_draw=args.max_boxes_to_draw,
      min_score_thresh=args.min_score_thresh,
      bm_runs=args.bm_runs,
      threads=args.threads,
      trace_filename=args.trace_filename,
      imagenes=img_list)


if __name__ == '__main__':
  # logging.set_verbosity(logging.WARNING)
  logging.set_verbosity(logging.INFO)
  tf.disable_eager_execution()
  app.run(main)
