# Kalman Filter from Scratch

Educational project to explore the Kalman filter in modern C++ (C++26).
No libraries: a fixed-size `Matrix` (compile-time shape checks, inverse, Cholesky) and a linear
`KalmanFilter`. A 2D tracking demo plus plots show how Q, R and P₀ change the filter's behavior.

## Build and test

```bash
cmake -S . -B build -DKF_WITH_KALMAN=ON
cmake --build build
ctest --test-dir build
```

## Run the demo

```bash
./build/kf_demo --help              # all options
./build/kf_demo > out.csv           # simulate and write CSV
```

## Plot

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once

.venv/bin/python tools/plot.py out.csv                        # dashboard for one run
.venv/bin/python tools/compare.py q 0.0005 0.05 1             # same world, different Q
.venv/bin/python tools/compare.py r 0.2 1 5 --turn 0          # different R
.venv/bin/python tools/compare.py q 0.0005 1 --animate live   # animation (or: --animate kf.mp4)
```
