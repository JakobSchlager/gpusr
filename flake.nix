{
  description = "GPuSR - GPU Genetic Programming for Symbolic Regression";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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

      cuda = pkgs.cudaPackages;

    in {
      devShells.${system}.default =
        (pkgs.mkShell.override {
          stdenv = cuda.backendStdenv;
        }) {
          packages = with pkgs; [
            cmake
            ninja
            gdb
            clang-tools
            pkg-config

            cuda.cudatoolkit

            python3
            python3Packages.numpy
          ];

          shellHook = ''
            export CUDA_PATH="${cuda.cudatoolkit}"
            export CUDAToolkit_ROOT="${cuda.cudatoolkit}"
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}"
          '';
        };
    };
}
