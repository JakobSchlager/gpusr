# gpusr

**GPU** Accelerated **S**ymbolic **R**egression using **G**enetic **P**rogramming.

## Structure and Building
- measurements/
  Contains the output of the sample application, including the best mse, best individual and runtime per file.
  These should not be considered as conclusive benchmark results. They should be viewed as rough estimates of each change's impact (they are commited alongside the   respective change).
- scripts/
  Contains a simple python script for dataset generation as well as a python notebook to visualize results.
- src/
- CMakeLists.txt
- build.sh
- flake.nix

## Building
Only nixos with the environment "flake.nix" above was tested.

```bash
cd gpusr
nix develop
./build.sh [cuda]
```

Note that depending on the target (cpu/ cuda) the evaluator has to be adapted in "main.cpp". *Note: Everything is very much a WIP. Currently, two compilation units are created (and compiled with gcc and nvcc respectively). This allows the usage of cpp-26 in the cpu path.*

TODO: Add a dev-container setup. Test on Windows.

