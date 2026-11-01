cmake -S . -B build/cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPUSR_ENABLE_CUDA=OFF

cmake --build build/cpu -j

ln -s build/cpu/compile_commands.json compile_commands.json
