#!/bin/sh

autopep8 --in-place --aggressive --aggressive $1
pylint $1
