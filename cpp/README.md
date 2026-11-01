# Small C++17 inspection stage

No OpenCV dependency in the C++ source/build. Google Test is downloaded by CMake
at an exact commit with a SHA-256 archive check. A native C++17 compiler, CMake
3.20+, and Internet access for that test dependency are needed. No Docker.

```sh
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/inspection_cli image.pgm 8 10 1
```

CLI arguments: binary P5 8-bit PGM path, residual threshold (default 8), minimum
component foreground-pixel count (default 10), optional timing repeat count
(default 1, maximum 100). PGM headers support comments; maxval must be 255.
Dimensions are limited to 4096×4096. Invalid/truncated inputs return exit code 2
and stderr. JSON stdout contains width, height, a row-major flat 0/255 `mask`,
`boxes` with integer x/y/width/height/pixels, and compute-only `processing_ms`.
The timing includes allocations and processing, excluding PGM/JSON input/output.

The implementation uses median 3×3 denoising (replicate borders), a separable
31×31 sigma=5 Gaussian (double precision, REFLECT_101 borders), absolute residual
strictly greater than the threshold, and handwritten eight-connected BFS. It has
no morphology and filters on foreground pixel count. This stage matches the
explicit Python/OpenCV reference in `camera_quality/cpp_bridge.py`, with one pixel
box tolerance and at most 0.1% mask disagreement in the generated-image parity
test. The benchmark records actual disagreement across all 180 test images.

The original Python inspector is preserved: it uses uint8 Gaussian rounding,
morphological close/open, contour area and centroids. Its boxes can differ from
this smaller stage. This C++ code is not a full port of that inspector and has
not been optimized; no runtime or speedup claim is implied by the language.

Python resolves `build/inspection_cli` (`.exe` on Windows), or the path in
`CPP_INSPECTOR`. On multi-configuration Windows builds use
`cmake --build build --config Release` and set `CPP_INSPECTOR` to the executable
in `build/Release`. MinGW users need the compiler runtime DLL directory on PATH.
