from __future__ import print_function

import os
import logging

#import tensorflow as tf
import numpy as np
import scipy.misc 
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x


def set_logger(log_path, log_name='training'):
  if log_path is None:
    print('log_path is empty')
    return None
    
  if os.path.exists(log_path):
    print('%s already exists'%log_path)
    return None

  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)

  logfile = logging.FileHandler(log_path)
  console = logging.StreamHandler()
  logfile.setLevel(logging.INFO)
  logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  console.setLevel(logging.DEBUG)
  console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(logfile)
  logger.addHandler(console)

  return logger


class TensorboardLogger(object):
  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    #self.writer = tf.compat.v1.summary.FileWriter(log_dir)
    self.writer = tf.summary.create_file_writer(log_dir)

  def scalar_summary(self, key, value, step):
    """Log a scalar variable."""
    # summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    #summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=key, simple_value=value)])
    #self.writer.add_summary(summary, step)
    with self.writer.as_default():
      tf.summary.scalar(key, value, step)
    self.writer.flush()

  def image_summary(self, key, images, step):
    """Log a list of images."""
    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")

      # Create an Image object
      img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tf.Summary.Value(tag='%s/%d' % (key, i), image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=img_summaries)
    self.writer.add_summary(summary, step)
    self.writer.flush()
        
  def histo_summary(self, key, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, histo=hist)])
    self.writer.add_summary(summary, step)
    self.writer.flush()
