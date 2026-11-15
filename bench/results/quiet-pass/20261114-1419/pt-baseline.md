| scene | renderer | ms/spp (min..max) | fit k | time to relMSE 1.00e-2 | time to relMSE 1.00e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|---|
| cornell | pt-megakernel | 13.9 (13.9..13.9) | 1.24 | 169 ms (168 ms..170 ms) | 1.1 s (1.1 s..1.1 s) | 7.25e-5 / 4.82e-4 @1024 spp |
| sponza-white | pt-megakernel | 49.3 (49.3..49.3) | 1.01 | ~34.4 s (extrapolated) (34.4 s..34.4 s) | not measurable (below 2x ref floor) | 1.44e-2 / 1.46e-2 @512 spp |

cornell / pt-megakernel: floor 4.39e-4, mean ratio vs Mitsuba 0.9999/0.9999/0.9999, relBias -0.01%, block-mean max/rms 0.3%/0.1%, fit r2 1.000 (3 pts), repeats 3, valid true, PLATEAU vs pbrt
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 65 ms (65 ms..66 ms) | 3.38e-2 | 3.46e-2
     16 | 233 ms (231 ms..233 ms) | 6.67e-3 | 7.16e-3
     64 | 897 ms (896 ms..899 ms) | 1.32e-3 | 1.74e-3
    256 | 3.6 s (3.6 s..3.6 s) | 2.71e-4 | 6.83e-4
   1024 | 14.2 s (14.2 s..14.2 s) | 7.25e-5 | 4.82e-4
  targets vs pbrt: 178 ms | 2.0 s
sponza-white / pt-megakernel: floor 4.09e-3, mean ratio vs Mitsuba 0.9987/0.9987/0.9988, relBias -0.13%, block-mean max/rms 4.9%/0.6%, fit r2 0.999 (5 pts), repeats 3, valid true
  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt
      4 | 207 ms (206 ms..207 ms) | 1.85e+0 | 1.85e+0
     16 | 799 ms (799 ms..799 ms) | 4.30e-1 | 4.32e-1
     64 | 3.2 s (3.2 s..3.2 s) | 1.05e-1 | 1.06e-1
    256 | 12.6 s (12.6 s..12.6 s) | 2.71e-2 | 2.73e-2
    512 | 25.3 s (25.3 s..25.3 s) | 1.44e-2 | 1.46e-2
  targets vs pbrt: ~34.8 s (extrapolated) | not measurable (below 2x ref floor)
appended 37 summary rows to /Users/dex/Developer/pathtracer/bench/results/competitor.csv (12 columns)
