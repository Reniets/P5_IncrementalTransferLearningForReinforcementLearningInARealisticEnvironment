#!/bin/sh
cd source

for j in 6
do
  python3 main.py $j 2
done