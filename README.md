# heightestimation
A spatial and temporal network (based of Peleg's in the paper below) combined in a two-stream network in Keras used to estimate height of a camera in egocentric video.  Temporal model based on model from paper below.

@inproceedings{poleg2016compact,
  title={Compact CNN for indexing egocentric videos},
  author={Poleg, Yair and Ephrat, Ariel and Peleg, Shmuel and Arora, Chetan},
  booktitle={2016 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1--9},
  year={2016},
  organization={IEEE}
}


Respective networks are in spatialnetwork.py, temporalnetwork.py, and twostream.py

Preprocessing is done in magereaders.py
