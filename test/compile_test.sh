g++ -std=c++17 -DNDEBUG -O3 \
    -I /home/p2-admin/include \
    -L /home/p2-admin/lib \
    gen_sdsl_vectors.cpp \
    -lsdsl -ldivsufsort -ldivsufsort64 \
    -o gen_sdsl_vectors
