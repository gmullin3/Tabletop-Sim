from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py


class TabletopBimanual(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'topview_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'leftview_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'rightview_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                        'ee_pos': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float64,
                            doc='2x robot ee pos',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float64,
                            doc='2x robot local joint',
                        ),
                        'delta_ee': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float64,
                            doc='2x robot ee delta',
                        ),
                        'delta_joint': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float64,
                            doc='2x robot joint delta',
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='../datasets/*/*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(hdf5_path):
            # load raw data --> this should change for your dataset
            root = h5py.File(hdf5_path, 'r')
            length =  root['/actions/ee_pos'].shape[0]

            ## Dataset is generated in 20Hz, I want to make it 5Hz.

            nli = root['/observations/instruction']
            top_image = root['/observations/images/top']
            leftview_image = root['/observations/images/left']
            rightview_image = root['/observations/images/right']

            joint_pos = root['/actions/joint_pos']
            ee_pos = root['/actions/ee_pos']
            delta_joint = root['/actions/joint_vel']
            delta_ee = root['/actions/ee_rpy_vel']

            ####### Important ################
            episode = []
            for i in range(length - shift):
                language_embedding = self._embed([nli[i]])[0].numpy()
                episode.append({
                        'observation': {
                            'topview_image': image[i],
                            'leftview_image': leftview_image[i],
                            'rightview_image': rightview_image[i],
                        },
                        'action': {
                            'ee_pos': ee_pos[i],
                            'joint_pos': joint_pos[i],
                            'delta_ee': delta_ee[i],
                            'delta_joint': delta_joint[i]
                        },
                        'language_instruction': nli[i],
                        'language_embedding': language_embedding,
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': hdf5_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return hdf5_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            print(sample)
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

