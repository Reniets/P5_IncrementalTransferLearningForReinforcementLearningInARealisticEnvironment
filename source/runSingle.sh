#!/bin/sh
cd source

for j in 3
do
  python3 main.py $j 0
done