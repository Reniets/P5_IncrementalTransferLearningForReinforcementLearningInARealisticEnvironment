#!/bin/sh
cd source

for j in 1 2 3 4 5 6
do
  python3 main.py $j 0
done

for j in 1 2 3 4 5
do
  python3 main.py $j 1
done