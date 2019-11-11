#!/bin/sh
cd source

for j in 3 4 5 6
do
  python3 main.py $j 2 true
done