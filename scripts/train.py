#!/usr/bin/env python3
"""
Training script for Kernel Tonic model with custom kernels and FP8 training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from training.trainer import main as train_main

if __name__ == "__main__":
    train_main() 