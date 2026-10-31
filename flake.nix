{
  description = "GPuSR - GPU Genetic Programming for Symbolic Regression";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "7.5" ];
        };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          cmake
          ninja
          gdb
          clang-tools
          pkg-config

          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          cudaPackages.cuda_cccl

          python3
          python3Packages.numpy
        ];

        shellHook = ''
          export CUDA_PATH="${pkgs.cudaPackages.cuda_nvcc}"
          export CUDAHOSTCXX="${pkgs.stdenv.cc}/bin/g++"
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"

          echo "GPuSR development environment"
          echo "GCC: $(g++ --version | head -n 1)"
          echo "CUDA: $(nvcc --version | tail -n 1)"
        '';
      };
    };
}
