#!/bin/bash
python3 -u plotter.py convnet_median1 &> convnet_median1_logs.out
python3 -u plotter.py convnet_log1 &> convnet_log1_logs.out
python3 -u plotter.py convnet_median_greyscale1 &> convnet_median_greyscale1_logs.out
python3 -u plotter.py convnet_bucket1 &> bucket1_logs.out
