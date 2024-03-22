#!/bin/bash -e


CUDA_VISIBLE_DEVICES="0" python3.10 dali_client2.py --id 0 --size 2 &
CUDA_VISIBLE_DEVICES="1" python3.10 dali_client2.py --id 1 --size 2 &