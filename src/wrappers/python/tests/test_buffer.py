#!/usr/bin/env python3

#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright Contributors to the OpenEXR Project.
#
# Test loading and storing from python buffers
#

from __future__ import print_function
import sys
import os
import tempfile
import atexit
import unittest
import numpy as np
from io import BytesIO

import OpenEXR

test_dir = os.path.dirname(__file__)
test_exr_path = f'{test_dir}/test.exr'


def compare_files(lhs, rhs):

    for Plhs, Prhs in zip(lhs.parts,rhs.parts):
        compare_parts(Plhs, Prhs)

def compare_parts(lhs, rhs):

    if len(lhs.channels) != len(rhs.channels):
        raise Exception(f"#channels in {lhs.name()} differs: {len(lhs.channels)} {len(rhs.channels)}")

    for c in lhs.channels.keys():
        compare_channels(lhs.channels[c], rhs.channels[c])

def compare_channels(lhs, rhs):

    if (lhs.name != rhs.name or
        lhs.type() != rhs.type() or
        lhs.xSampling != rhs.xSampling or
        lhs.ySampling != rhs.ySampling):
        raise Exception(f"channel {lhs.name} differs: {lhs.__repr__()} {rhs.__repr__()}")

    compare_channel_pixels(lhs, rhs)
    
def compare_channel_pixels(lhs, rhs):

    if lhs.pixels.shape != rhs.pixels.shape:
        raise Exception(f"channel {lhs.name}: image size differs: {lhs.pixels.shape} vs. {rhs.pixels.shape}")
        
    height = lhs.pixels.shape[0]
    width = lhs.pixels.shape[1]
    for y in range(height):
        for x in range(width):
            l = lhs.pixels[y,x]
            r = rhs.pixels[y,x]
            if l is None and r is None:
                continue
            close = np.isclose(l, r, 1e-5)
            if not np.all(close):
                for i in np.argwhere(close==False):
                    y,x = i
                    if math.isfinite(lhs.pixels[y,x]) and math.isfinite(rhs.pixels[y,x]):
                        raise Exception(f"channel {lhs.name}: deep pixels {i} differ: {lhs.pixels[y,x]} {rhs.pixels[y,x]}")

class TestBuffer(unittest.TestCase):

    def setUp(self):
        # Print the name of the current test method
        print(f"Running test {self.id().split('.')[-1]}")


    def test_buffer_read_rgba(self):
        # load EXR from file and from buffer, compare
        fileEXR = OpenEXR.File(test_exr_path)

        bufferEXR = None
        with open(test_exr_path, 'rb') as file:
            bufferEXR = OpenEXR.File(file.read())

        compare_files(fileEXR, bufferEXR)
        
    
    def test_buffered_write_rgba(self):
        # write bytes to buffered writer instead of file
        fileEXR = OpenEXR.File(test_exr_path)
        buffered = BytesIO()
        fileEXR.write(buffered)

        # assert that we can load from buffer again
        bufferEXR = OpenEXR.File(buffered.getvalue()) # alternatively buffer.seek(0) and then buffer.read()
        compare_files(fileEXR, bufferEXR)


    def test_buffer_write_rgba(self):
        # write bytes to allocated buffer instead of file
        fileEXR = OpenEXR.File(test_exr_path)
        buffer = np.empty(1 << 22, dtype=np.uint8)  # allocate 4MiB
        bytes_written = fileEXR.write(buffer)

        # assert that we can load from buffer again
        bufferEXR = OpenEXR.File(buffer)
        compare_files(fileEXR, bufferEXR)


    def test_buffered_chain_write_rgba(self):
        # write to and load from concatenated byte stream
        fileEXR = OpenEXR.File(test_exr_path)
        buffer = np.empty(1 << 22, dtype=np.uint8)  # allocate 4MiB
        buf_size = fileEXR.write(buffer)

        # concatenate another file to stream
        fileEXR.write(buffer[buf_size:])

        # assert that we can load from buffer again
        # load first file
        buffer1EXR = OpenEXR.File(buffer)
        compare_files(fileEXR, buffer1EXR)
        
        # load second file
        buffer2EXR = OpenEXR.File(buffer[buf_size:])
        compare_files(buffer1EXR, buffer2EXR)


    def test_buffer_chain_write_rgba(self):
        # write to and load from concatenated byte stream
        fileEXR = OpenEXR.File(test_exr_path)
        buffer = BytesIO()
        buf_size = fileEXR.write(buffer)
        assert len(buffer.getvalue()) == buf_size

        # concatenate another file to stream
        fileEXR.write(buffer)
        assert len(buffer.getvalue()) == 2*buf_size

        # assert that we can load from buffer again
        buffer.seek(0)
        # load first file
        buffer1EXR = OpenEXR.File(buffer.read(buf_size))
        compare_files(fileEXR, buffer1EXR)
        
        # load second file
        buffer2EXR = OpenEXR.File(buffer.read(buf_size))
        compare_files(buffer1EXR, buffer2EXR)

        assert buffer.tell() == 2*buf_size


    def test_invalid_types(self):
        
        with self.assertRaises(RuntimeError):
            f = OpenEXR.File(b'invalid buffer content')
        
        # valid type, invalid content
        with self.assertRaises(RuntimeError):
            f = OpenEXR.File(np.arange(100))

        # invalid filetype
        with self.assertRaises(TypeError):
            f = OpenEXR.File(BytesIO)

        fileEXR = OpenEXR.File(test_exr_path)
        # buffer size too small
        buffer = np.empty(1 << 4, dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            fileEXR.write(buffer)
        
        # no exclusive write access possible (SHARED)
        buffer = BytesIO()
        mem    = buffer.getbuffer() # locks the underlying data structure
        with self.assertRaises(BufferError):
            fileEXR.write(buffer)

        # writing must work again after releasing lock
        # SHARED -> EXCLUSIVE
        del mem
        fileEXR.write(buffer)
        

if __name__ == '__main__':
    unittest.main()
    print("OK")


