cd build/
make all -j4
./compose.bin --net_resolution 320x176 --render_pose 1 --model_pose MPI_4_layers
