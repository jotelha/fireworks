#!/bin/bash
# on bwCloud image jlh-bwcloud, install or update with
sudo python setup.py --verbose install --prefix=/usr/local
# and a subsequent
sudo chmod -R a+rX /usr/local
# to be on the safe side