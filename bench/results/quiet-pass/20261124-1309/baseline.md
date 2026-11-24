| scene | renderer | ms/spp (min..max) | fit k | time to relMSE 1.00e-2 | time to relMSE 1.00e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|---|
| cornell | tgpt-wavefront | 12.1 (12.0..12.2) | 1.12 | 206 ms (203 ms..207 ms) | 1.8 s (1.7 s..1.8 s) | 1.53e-4 / 5.64e-4 @1024 spp |
| cornell | tgpt-megakernel | 63.9 (62.7..65.1) | 1.12 | 1.1 s (1.1 s..1.2 s) | 9.0 s (9.0 s..9.4 s) | 1.38e-4 / 5.48e-4 @1024 spp |
| cornell | mutracer | 6.57 (6.5..6.7) | 1.12 | 106 ms (105 ms..106 ms) | 863 ms (862 ms..865 ms) | 1.38e-4 / 5.44e-4 @1024 spp |
| sponza-white | tgpt-wavefront | 118.3 (114.0..118.9) | 0.97 | ~1.0 min (extrapolated) (1.0 min..1.1 min) | not measurable (below 2x ref floor) | 1.08e-2 / 1.10e-2 @512 spp |
| sponza-white | tgpt-megakernel | 426.3 (424.8..429.2) | 0.99 | 3.1 min (3.1 min..3.1 min) | not measurable (below 2x ref floor) | 8.72e-3 / 8.88e-3 @512 spp |
| sponza-white | mutracer | 70.8 (66.6..71.4) | 0.89 | ~1.4 min (extrapolated) (59.0 s..1.4 min) | not measurable (below 2x ref floor) | 1.72e-2 / 1.74e-2 @512 spp |

cornell / tgpt-wavefront: floor 4.39e-4, mean ratio vs Mitsuba 0.9999/0.9999/0.9999, relBias -0.01%, block-mean max/rms 0.5%/0.1%, fit r2 0.999 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 111 ms (107 ms..111 ms) | 2.08e-2 | 2.14e-2
     16 | 283 ms (279 ms..284 ms) | 6.87e-3 | 7.36e-3
     64 | 982 ms (968 ms..988 ms) | 1.78e-3 | 2.22e-3
    256 | 4.0 s (3.9 s..4.0 s) | 4.48e-4 | 8.62e-4
   1024 | 12.4 s (12.3 s..12.4 s) | 1.53e-4 | 5.64e-4
  targets vs pbrt: 216 ms | 3.2 s
cornell / tgpt-megakernel: floor 4.39e-4, mean ratio vs Mitsuba 0.9999/0.9999/0.9999, relBias -0.01%, block-mean max/rms 0.4%/0.1%, fit r2 1.000 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 253 ms (252 ms..265 ms) | 5.25e-2 | 5.35e-2
     16 | 1.0 s (1.0 s..1.1 s) | 1.10e-2 | 1.15e-2
     64 | 4.1 s (4.1 s..4.3 s) | 2.34e-3 | 2.78e-3
    256 | 16.3 s (16.2 s..17.0 s) | 5.33e-4 | 9.48e-4
   1024 | 1.1 min (1.1 min..1.1 min) | 1.38e-4 | 5.48e-4
  targets vs pbrt: 1.2 s | 15.2 s
cornell / mutracer: floor 4.39e-4, mean ratio vs Mitsuba 0.9994/0.9997/0.9998, relBias -0.04%, block-mean max/rms 1.1%/0.2%, fit r2 1.000 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 25 ms (25 ms..26 ms) | 4.94e-2 | 5.04e-2
     16 | 101 ms (101 ms..101 ms) | 1.05e-2 | 1.11e-2
     64 | 396 ms (396 ms..397 ms) | 2.29e-3 | 2.72e-3
    256 | 1.6 s (1.6 s..1.6 s) | 5.22e-4 | 9.34e-4
   1024 | 6.7 s (6.6 s..6.9 s) | 1.38e-4 | 5.44e-4
  targets vs pbrt: 111 ms | 1.5 s
sponza-white / tgpt-wavefront: floor 4.09e-3, mean ratio vs Mitsuba 0.9982/0.9981/0.9984, relBias -0.18%, block-mean max/rms 2.7%/0.6%, fit r2 1.000 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 508 ms (477 ms..509 ms) | 1.11e+0 | 1.11e+0
     16 | 1.9 s (1.8 s..1.9 s) | 2.98e-1 | 2.99e-1
     64 | 7.6 s (7.2 s..7.7 s) | 7.45e-2 | 7.48e-2
    256 | 30.6 s (29.5 s..31.0 s) | 1.96e-2 | 1.98e-2
    512 | 1.0 min (58.4 s..1.0 min) | 1.08e-2 | 1.10e-2
  targets vs pbrt: ~1.1 min (extrapolated) | not measurable (below 2x ref floor)
sponza-white / tgpt-megakernel: floor 4.09e-3, mean ratio vs Mitsuba 0.9982/0.9981/0.9984, relBias -0.18%, block-mean max/rms 2.6%/0.5%, fit r2 0.999 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 1.7 s (1.7 s..1.8 s) | 1.06e+0 | 1.06e+0
     16 | 6.7 s (6.6 s..7.0 s) | 2.51e-1 | 2.52e-1
     64 | 26.7 s (26.5 s..27.5 s) | 5.98e-2 | 6.00e-2
    256 | 1.8 min (1.8 min..1.8 min) | 1.57e-2 | 1.59e-2
    512 | 3.6 min (3.6 min..3.7 min) | 8.72e-3 | 8.88e-3
  targets vs pbrt: 3.1 min | not measurable (below 2x ref floor)
sponza-white / mutracer: floor 4.09e-3, mean ratio vs Mitsuba 0.9980/0.9981/0.9983, relBias -0.19%, block-mean max/rms 6.5%/0.7%, fit r2 0.988 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 232 ms (231 ms..856 ms) | 2.18e+0 | 2.18e+0
     16 | 924 ms (924 ms..3.2 s) | 5.15e-1 | 5.16e-1
     64 | 4.3 s (4.3 s..12.1 s) | 1.27e-1 | 1.28e-1
    256 | 32.2 s (17.1 s..32.3 s) | 3.27e-2 | 3.29e-2
    512 | 36.3 s (34.1 s..36.6 s) | 1.72e-2 | 1.74e-2
  targets vs pbrt: ~1.4 min (extrapolated) | not measurable (below 2x ref floor)
appended 111 summary rows to /Users/dex/Developer/pathtracer/bench/results/competitor.csv (12 columns)
