#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 00:18:36 2024

@author: Angel Ayala
"""

from torch.utils.tensorboard import SummaryWriter

writer = None
step = 0

def summary_create(path):
    global writer
    writer = SummaryWriter(path)
    return writer

def summary():
    global writer
    return writer

def summary_scalar(tag, value):
    global writer, step
    if writer:
        writer.add_scalar(tag, value, step)

def summary_image(tag, value):
    global writer, step
    if writer:
        writer.add_image(tag, value, step)

def summary_step(step_val):
    global step
    step = step_val
