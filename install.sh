#!/bin/bash

#rm *.so
python setup.py build_ext -i
mv *.so ../../
