# IREE Runtime Hello World with CMake

[![Build with Latest IREE Release](https://github.com/iree-org/iree-template-runtime-cmake/actions/workflows/build-and-test.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree-template-runtime-cmake/actions/workflows/build-and-test.yml?query=branch%3Amain+event%3Apush)

## Instructions

### Cloning the Repository

Use GitHub's "Use this template" feature to create a new repository or clone it
manually:

```sh
$ git clone https://github.com/iree-org/iree-template-runtime-cmake.git
$ cd iree-template-runtime-cmake
$ git submodule update --init --recursive
```

The only requirement is that the main IREE repository is added as a submodule.
If working in an existing repository then add the submodule and ensure it has
its submodules initialized:

```sh
$ git submodule add https://github.com/openxla/iree.git third_party/iree/
$ git submodule update --init --recursive
```

For a faster checkout some compiler-only dependencies can be dropped as this
template is only compiling the runtime (only bother if optimizing build bots):

```sh
$ git \
    -c submodule."third_party/llvm-project".update=none \
    -c submodule."third_party/stablehlo".update=none \
    -c submodule."third_party/torch-mlir".update=none \
    submodule update --init --recursive
```

### Building the Runtime

The [CMakeLists.txt](./CMakeLists.txt) adds the IREE CMake files as a subproject
and configures it for runtime-only compilation. A project wanting to build the
compiler from source or include other HAL drivers (CUDA, Vulkan, multi-threaded
CPU, etc) can change which options they set before they `add_subdirectory` on
the IREE project and/or pass the configuration to the CMake configure command.

The sample currently compiles in the synchronous CPU HAL driver (`local-sync`).

```sh
$ cmake -B build/ -GNinja .
$ cmake --build build/ --target hello_world
```

### Compiling the Sample Module

This sample assumes that the latest IREE compiler release is installed and used
to compile the module. For many users upgrading their `iree-compiler` install
when they bump their submodule should be sufficient to ensure the compiler and
runtime are compatible. In the future the compiler and runtime will have more
support for version shifting.

The sample currently assumes a CPU HAL driver and only produces a VMFB
supporting that. Additional compiler options can be used to change the target
HAL driver, target device architecture, etc. IREE supports multi-targeting both
across device types (CPU/GPU/etc) and architectures (AArch64/x86-64/etc) but the
command line interfaces are still under development. Basic CPU cross-compiling
can be accomplished with the `--iree-llvm-target-triple=` flag specifying the
CPU architecture.

```sh
$ python -m pip install iree-compiler --upgrade --user
$ iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=x86_64 \
    simple_mul.mlir \
    -o build/simple_mul.vmfb
```

### Running the Sample

The included sample program takes the device URI (in this case `local-sync`) and
compiled module file (`simple_mul.vmfb` as output from above) and prints the
output of a simple calculation. More advanced features like asynchronous
execution, providing output storage buffers for results, and stateful programs
are covered in other IREE samples.

```sh
$ ./build/hello_world local-sync build/simple_mul.vmfb
4xf32=1 1.1 1.2 1.3
 *
4xf32=10 100 1000 10000
 =
4xf32=10 110 1200 13000
```
