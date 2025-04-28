from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import h5py


class AlohaShoesTable(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'agentview_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(240, 320, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state':tfds.features.Tensor(
                            shape=(20,),
                            dtype=np.float32,
                            doc='xyz, 6D, grp',
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                        'ee_6d_pos': tfds.features.Tensor(
                            shape=(20,),
                            dtype=np.float32,
                            doc='2x robot ee pos',
                        ),
                        'ee_quat_pos': tfds.features.Tensor(
                            shape=(16,),
                            dtype=np.float32,
                            doc='2x robot ee pos',
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
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
            'train': self._generate_examples(path='/data5/jellyho/tabletop/aloha_shoes_table/*.hdf5'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(hdf5_path):
            # load raw data --> this should change for your dataset
            root = h5py.File(hdf5_path, 'r')
            length =  root['/actions/ee_6d_pos'].shape[0]
            nli = root['/observations/states/language_instruction']
            agentview_image = root['/observations/images/back']
            leftwrist_image = root['/observations/images/wrist_left']
            rightwrist_image = root['/observations/images/wrist_right']
            states = root['/observations/states/ee_6d_pos']

            ee_6d_pos = root['/actions/ee_6d_pos']
            ee_quat_pos = root['/actions/ee_quat_pos']

            ####### Important ################
            episode = []
            for i in range(length):
                episode.append({
                        'observation': {
                            'agentview_image': agentview_image[i],
                            'left_wrist_image': leftwrist_image[i],
                            'right_wrist_image': rightwrist_image[i],
                            'state': states[i]
                        },
                        'action': {
                            'ee_6d_pos': ee_6d_pos[i],
                            'ee_quat_pos': ee_quat_pos[i],
                        },
                        'language_instruction': nli[i],
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

