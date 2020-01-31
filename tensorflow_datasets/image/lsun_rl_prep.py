"""LSUN room layout dataset. This one is preprocessed.

Large scene understanding dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_IMAGE_URL = "https://web.archive.org/web/20170221104221/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/image.zip"
_TRAIN_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/training.mat"
_VALID_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/challenge/2015/roomlayout/data/validation.mat"
_DOC_URL = "https://web.archive.org/web/20160617045914/http://lsun.cs.princeton.edu/2016/"

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




class LsunRLPrep(tfds.core.GeneratorBasedBuilder):
  """Lsun Room Layout Estimation dataset, preprocessed (cropping and image augmentation)

  """
  MANUAL_DOWNLOAD_INSTRUCTIONS = \
  """
    The data directory should have the following structure:
  ```
  path/to/manual_dir/lsun_rl_prep/images_v{VERSION}/
    images_preproc_edge_train/  # Preprocessed edge images for training
        xxx.png
        xxy.png
        xxa.png
    images_preproc_edge_val/  # Preprocessed edge images for validation
        xxx.png
        xxy.png
        xxa.png
    images_preproc_orig_train/  # Preprocessed input images for training
        xxx.png
        xxy.png
        xxa.png
    images_preproc_orig_val/  # Preprocessed input images for validation
        xxx.png
        xxy.png
        xxa.png
  ```

  To use it:
  ```
  dl_config = tfds.download.DownloadConfig(manual_dir='path/to/manual_dir/')
  tfds.load(
      'lsun_rl_prep',
      split='split_name'
      download_and_prepare_kwargs=dict(download_config=dl_config),
  )
  ```
  """

  VERSION = tfds.core.Version('0.1.0', 'Images resized to 320x320')
  SUPPORTED_VERSIONS = [
      tfds.core.Version(
          '0.1.1', 'Images resized to 240x320'),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="png"),
            "edges": tfds.features.Image(encoding_format="png"),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=('image','edges'),
        # Homepage of the dataset for documentation
        homepage=_DOC_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    print(dl_manager.manual_dir)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "image_dir": os.path.join(dl_manager.manual_dir, 'images_v{}/images_preproc_orig_val/'.format(self.version)),
                "edges_dir": os.path.join(dl_manager.manual_dir, 'images_v{}/images_preproc_edge_val/'.format(self.version))
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_dir": os.path.join(dl_manager.manual_dir,'images_v{}/images_preproc_orig_train/'.format(self.version)),
                "edges_dir": os.path.join(dl_manager.manual_dir,'images_v{}/images_preproc_edge_train/'.format(self.version))
            },
        ),

    ]

  def _generate_examples(self, image_dir, edges_dir):
    """Yields examples."""

    for fname in tf.io.gfile.listdir(image_dir):
      image_full_name = os.path.join(image_dir,fname)
      edges_full_name = os.path.join(edges_dir,fname)

      record = {
        'image': image_full_name,
        'edges': edges_full_name,
      }

      yield fname, record
