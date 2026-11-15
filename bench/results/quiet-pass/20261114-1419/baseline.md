| scene | renderer | ms/spp (min..max) | fit k | time to relMSE 1.00e-2 | time to relMSE 1.00e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|---|
| cornell | tgpt-wavefront | 11.5 (11.3..12.4) | 1.13 | 215 ms (215 ms..218 ms) | 1.8 s (1.8 s..1.8 s) | 1.53e-4 / 5.64e-4 @1024 spp |
| cornell | tgpt-megakernel | 66.2 (64.4..66.4) | 1.12 | 1.1 s (1.1 s..1.1 s) | 9.2 s (9.1 s..9.3 s) | 1.38e-4 / 5.48e-4 @1024 spp |
| cornell | mutracer | 6.63 (6.5..6.7) | 1.12 | 114 ms (110 ms..114 ms) | 942 ms (894 ms..945 ms) | 1.38e-4 / 5.44e-4 @1024 spp |
| sponza-white | tgpt-wavefront | 105.7 (104.1..106.8) | 0.97 | ~56.0 s (extrapolated) (55.5 s..56.4 s) | not measurable (below 2x ref floor) | 1.08e-2 / 1.10e-2 @512 spp |
| sponza-white | tgpt-megakernel | 379.6 (347.9..414.2) | 1.00 | 2.8 min (2.5 min..3.0 min) | not measurable (below 2x ref floor) | 8.72e-3 / 8.88e-3 @512 spp |
| sponza-white | mutracer | 57.7 (57.3..57.8) | 1.00 | ~48.7 s (extrapolated) (48.5 s..48.7 s) | not measurable (below 2x ref floor) | 1.72e-2 / 1.74e-2 @512 spp |

cornell / tgpt-wavefront: floor 4.39e-4, mean ratio vs Mitsuba 0.9999/0.9999/0.9999, relBias -0.01%, block-mean max/rms 0.5%/0.1%, fit r2 0.999 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 116 ms (113 ms..117 ms) | 2.08e-2 | 2.14e-2
     16 | 295 ms (294 ms..299 ms) | 6.87e-3 | 7.36e-3
     64 | 1.0 s (1.0 s..1.0 s) | 1.79e-3 | 2.22e-3
    256 | 3.9 s (3.8 s..3.9 s) | 4.48e-4 | 8.61e-4
   1024 | 11.8 s (11.6 s..12.7 s) | 1.53e-4 | 5.64e-4
  targets vs pbrt: 226 ms | 3.1 s
cornell / tgpt-megakernel: floor 4.39e-4, mean ratio vs Mitsuba 0.9999/0.9999/0.9999, relBias -0.01%, block-mean max/rms 0.4%/0.1%, fit r2 1.000 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 253 ms (253 ms..262 ms) | 5.25e-2 | 5.35e-2
     16 | 1.0 s (1.0 s..1.0 s) | 1.10e-2 | 1.15e-2
     64 | 4.1 s (4.1 s..4.1 s) | 2.34e-3 | 2.78e-3
    256 | 16.8 s (16.5 s..16.9 s) | 5.33e-4 | 9.48e-4
   1024 | 1.1 min (1.1 min..1.1 min) | 1.38e-4 | 5.48e-4
  targets vs pbrt: 1.2 s | 15.6 s
cornell / mutracer: floor 4.39e-4, mean ratio vs Mitsuba 0.9994/0.9997/0.9998, relBias -0.04%, block-mean max/rms 1.1%/0.2%, fit r2 1.000 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 28 ms (28 ms..29 ms) | 4.94e-2 | 5.04e-2
     16 | 109 ms (105 ms..109 ms) | 1.05e-2 | 1.11e-2
     64 | 433 ms (410 ms..435 ms) | 2.29e-3 | 2.72e-3
    256 | 1.7 s (1.6 s..1.7 s) | 5.22e-4 | 9.34e-4
   1024 | 6.8 s (6.7 s..6.9 s) | 1.38e-4 | 5.44e-4
  targets vs pbrt: 120 ms | 1.6 s
sponza-white / tgpt-wavefront: floor 4.09e-3, mean ratio vs Mitsuba 0.9982/0.9981/0.9984, relBias -0.18%, block-mean max/rms 2.7%/0.6%, fit r2 1.000 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 456 ms (447 ms..467 ms) | 1.11e+0 | 1.11e+0
     16 | 1.7 s (1.7 s..1.7 s) | 2.98e-1 | 2.99e-1
     64 | 6.8 s (6.8 s..6.9 s) | 7.45e-2 | 7.48e-2
    256 | 27.4 s (27.1 s..27.6 s) | 1.96e-2 | 1.98e-2
    512 | 54.1 s (53.3 s..54.7 s) | 1.08e-2 | 1.10e-2
  targets vs pbrt: ~56.7 s (extrapolated) | not measurable (below 2x ref floor)
sponza-white / tgpt-megakernel: floor 4.09e-3, mean ratio vs Mitsuba 0.9982/0.9981/0.9984, relBias -0.18%, block-mean max/rms 2.6%/0.5%, fit r2 0.999 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 1.6 s (1.4 s..1.6 s) | 1.06e+0 | 1.06e+0
     16 | 6.4 s (5.6 s..6.6 s) | 2.51e-1 | 2.52e-1
     64 | 25.4 s (22.5 s..26.1 s) | 5.98e-2 | 6.00e-2
    256 | 1.7 min (1.5 min..1.8 min) | 1.57e-2 | 1.59e-2
    512 | 3.2 min (3.0 min..3.5 min) | 8.72e-3 | 8.88e-3
  targets vs pbrt: 2.8 min | not measurable (below 2x ref floor)
sponza-white / mutracer: floor 4.09e-3, mean ratio vs Mitsuba 0.9980/0.9981/0.9983, relBias -0.19%, block-mean max/rms 6.5%/0.7%, fit r2 1.000 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 231 ms (231 ms..231 ms) | 2.18e+0 | 2.18e+0
     16 | 928 ms (920 ms..929 ms) | 5.15e-1 | 5.16e-1
     64 | 3.7 s (3.7 s..3.7 s) | 1.27e-1 | 1.28e-1
    256 | 14.7 s (14.7 s..14.7 s) | 3.27e-2 | 3.29e-2
    512 | 29.5 s (29.4 s..29.6 s) | 1.72e-2 | 1.74e-2
  targets vs pbrt: ~49.2 s (extrapolated) | not measurable (below 2x ref floor)
appended 111 summary rows to /Users/dex/Developer/pathtracer/bench/results/competitor.csv (12 columns)
