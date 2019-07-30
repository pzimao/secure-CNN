#!/bin/bash
g++ -std=c++14 $1 -o $2 -I /home/liufei/Documents/20190310/cnn/include -L /usr/local/lib/libboost_system.so.1.65.1 /usr/lib/x86_64-linux-gnu/libboost_system.so.1.58.0  /home/liufei/Documents/20190310/cnn/libs/libgazelle.so -lboost_thread
