all: libbuild_matrix.so

cbits/build_matrix.o: cbits/build_matrix.c
	$(CC) -Wall -Wextra -pedantic -march=nocona -mtune=haswell -fvisibility=hidden -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -DNDEBUG -ffunction-sections $(CFLAGS) -c cbits/build_matrix.c -o cbits/build_matrix.o

libbuild_matrix.so: cbits/build_matrix.o
	$(CC) -Wl,--gc-sections $(LDFLAGS) -shared cbits/build_matrix.o -o annealing_sign_problem/libbuild_matrix.so

.PHONY: clean
clean:
	rm -f cbits/build_matrix.o annealing_sign_problem/libbuild_matrix.so
