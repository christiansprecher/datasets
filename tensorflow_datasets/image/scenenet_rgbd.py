"""SceneNet_RGBD

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_CITATION = """\
@article{McCormac:etal:arXiv2016,
        author = {John McCormac and
            Ankur Handa and
                  Stefan Leutenegger and
                  Andrew J.Davison},
        title = {SceneNet RGB-D: 5M Photorealistic Images of Synthetic Indoor Trajectories with Ground Truth },
          booktitle = {arXiv},
        year = {2016}
}
"""

_DESCRIPTION = """
We introduce SceneNet RGB-D, expanding the previous work of SceneNet to enable large scale 
photorealistic rendering of indoor scene trajectories. It provides pixel-perfect ground 
truth for scene understanding problems such as semantic segmentation, instance segmentation, 
and object detection, and also for geometric computer vision problems such as optical flow, 
depth estimation, camera pose estimation, and 3D reconstruction.
"""

_DOC_URL = 'https://robotvault.bitbucket.io/scenenet-rgbd.html?'




class SceneNetRGBD(tfds.core.GeneratorBasedBuilder):
  """SceneNet RGB-D features photorealistic renderings
  """
  MANUAL_DOWNLOAD_INSTRUCTIONS = \
  """
  ```
  path/to/manual_dir/tf_images/
    train_edges/  # Preprocessed edge images for training
        xxx.png
        xxy.png
        xxa.png
    val_edges/  # Preprocessed edge images for validation
        xxx.png
        xxy.png
        xxa.png
    train_images/  # Preprocessed input images for training
        xxx.png
        xxy.png
        xxa.png
    val_images/  # Preprocessed input images for validation
        xxx.png
        xxy.png
        xxa.png
  ```

  To use it:
  ```
  dl_config = tfds.download.DownloadConfig(manual_dir='path/to/manual_dir/')
  tfds.load(
      'scene_net_rgbd',
      split='split_name'
      download_and_prepare_kwargs=dict(download_config=dl_config),
  )
  ```
  """

  VERSION = tfds.core.Version('0.1.0',)

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="jpeg"),
            "edges": tfds.features.Image(encoding_format="jpeg"),
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

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "image_dir": os.path.join(dl_manager.manual_dir, 'tf_images/val_images/'),
                "edges_dir": os.path.join(dl_manager.manual_dir, 'tf_images/val_edges/')
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_dir": os.path.join(dl_manager.manual_dir,'tf_images/train_images/'),
                "edges_dir": os.path.join(dl_manager.manual_dir,'tf_images/train_edges/')
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
