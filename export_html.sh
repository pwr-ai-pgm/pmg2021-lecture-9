#!/bin/bash

docker run --rm -it -v $PWD:/lecture pmg2021-lecture-9 jupyter nbconvert --to slides ./Lecture.ipynb