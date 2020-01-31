"""LSUN room layout dataset.

Large scene understanding dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tensorflow as tf
import scipy.io as sio
import tensorflow_datasets.public_api as tfds

_IMAGE_URL = "https://web.archive.org/web/20170221104221/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/image.zip"
_TRAIN_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat"
_VALID_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat"
_DOC_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/2016/"

_NUM_CLASSES = 11

_CITATION = """\
@article{journals/corr/YuZSSX15,
  added-at = {2018-08-13T00:00:00.000+0200},
  author = {Yu, Fisher and Zhang, Yinda and Song, Shuran and Seff, Ari and Xiao, Jianxiong},
  biburl = {https://www.bibsonomy.org/bibtex/2446d4ffb99a5d7d2ab6e5417a12e195f/dblp},
  ee = {http://arxiv.org/abs/1506.03365},
  interhash = {3e9306c4ce2ead125f3b2ab0e25adc85},
  intrahash = {446d4ffb99a5d7d2ab6e5417a12e195f},
  journal = {CoRR},
  keywords = {dblp},
  timestamp = {2018-08-14T15:08:59.000+0200},
  title = {LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop.},
  url = {http://dblp.uni-trier.de/db/journals/corr/corr1506.html#YuZSSX15},
  volume = {abs/1506.03365},
  year = 2015
}
"""

_DESCRIPTION = """
The room layout estimation is formulated as a way to predict the positions 
of intersection between planar walls, ceiling and floors. There are 4000 
images for training, 394 images for validation and 1000 images for testing.
"""




class LsunRoomLayout(tfds.core.GeneratorBasedBuilder):
  """Lsun Room Layout Estimation dataset

  """
  MANUAL_DOWNLOAD_INSTRUCTIONS = \
  """  
  ```
  path/to/manual_dir/lsun_room_layout/
    images/  # All images
        xxx.jpg
        xxy.jpg
        xxa.jpg
    training/ # All training .mat files
        xxx.mat
        xxy.mat
        xxz.mat
    validation/ # All validation .mat files
        xxa.mat
        xxb.mat
        xxc.mat
  ```

  To use it:
  ```
  dl_config = tfds.download.DownloadConfig(manual_dir='path/to/manual_dir/')
  tfds.load(
      'lsun_room_layout',
      split='split_name'
      download_and_prepare_kwargs=dict(download_config=dl_config),
  )
  ```
  """

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="jpeg"),
            "layout": tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
            "corners": tfds.features.Tensor(shape=(None,2), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=None,
        # Homepage of the dataset for documentation
        homepage=_DOC_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "mat_dir": os.path.join(dl_manager.manual_dir, "validation/"),
                "image_dir": os.path.join(dl_manager.manual_dir, "images/")
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "mat_dir": os.path.join(dl_manager.manual_dir, "training/"),
                "image_dir": os.path.join(dl_manager.manual_dir, "images/")
            },
        ),

    ]

  def _generate_examples(self, mat_dir, image_dir):
    """Yields examples."""

    for fname in tf.io.gfile.listdir(mat_dir):
      mat_full = os.path.join(mat_dir,fname)
      image, corners, label  = _load_mat_files(mat_full)

      # tf.print(corners.dtype)

      image_full = os.path.join(image_dir, image) + '.jpg'
      record = {
        'image': image_full,
        'corners': corners,
        'layout': label,
      }

      yield image, record


def _load_mat_files(file_name):
  mat = sio.loadmat(file_name)['a'][0]

  image = mat['image'][0][0]
  corners = tf.convert_to_tensor(mat['point'][0], dtype=tf.float64)
  label = mat['type'][0][0][0]

  return image, corners, label
