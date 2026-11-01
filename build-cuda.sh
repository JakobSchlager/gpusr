cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPUSR_ENABLE_CUDA=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build/cuda -j

ln -s build/cuda/compile_commands.json compile_commands.json
