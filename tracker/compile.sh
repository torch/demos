#!/bin/bash

# source (http://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html)

gcc -c -Wall -Werror -fpic fastdist.c
gcc -shared -o libfastdist.so fastdist.o

## install in default path
#
#cp $PWD/libxillyconv.so /usr/lib
#chmod 0755 /usr/lib/libxillyconv.so
#ldconfig
#gcc -Wall -o test xillyconv-test.c -lxillyconv
#./test
