enable_cuda="OFF"
build_subdir="cpu"

if [$1 == "cuda"]; then 
  enable_cuda="ON"
  build_subdir="cuda"
fi

cmake -S . -B build/cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPUSR_ENABLE_CUDA=$enable_cuda \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build/$build_subdir -j

rm compile_commands.json
ln -s build/$build_subdir/compile_commands.json compile_commands.json
