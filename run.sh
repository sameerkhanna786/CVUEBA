#!/bin/bash

python3 split_dataset.py
python3 generate_images.py
python3 exec_aug.py
python3 CVUEBA.py