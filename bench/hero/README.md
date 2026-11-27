# Neural radiance cache interior demo

Run the following from the pathtracer root with the usual bench dependencies installed. Asset preparation requires Python 3, Node 22.18+ and Blender (tested with 5.1.1). The preparation command downloads and verifies the original scenes and color maps, converts the materials and generates the browser assets locally. Pass `--blender <executable>` if Blender is outside the default search path. Preparing just `classroom` or `sponza` also works.

```sh
python3 bench/hero/prepare.py
node bench/live.ts nrc-hero
```

The page opens on Classroom in a WebGPU browser. It trains the compact fp16 MLP continuously at 960 × 540 (fp32 fallback), traces one bounce before querying the cache, denoises color with SVGF and displays the temporal upscaler at 1920 × 1080. On a slower GPU choose 640 × 360 under **Render controls** or open the page with `?res=640x360`. Startup uses the pipeline's short SVGF training gate. Choose **Plain PT / NRC** for a split with identical reconstruction on both sides. **Pipeline stage** switches between raw color, denoised color, and final output using the current camera and cache. For a presentation, choose **Fullscreen**, then **Render controls** to adjust daylight over the live image. Close the panel for a clear view and choose **Exit fullscreen** to return. Shift-drag moves the divider. Drag to look, use WASD or the arrow keys to move, Q/E for height and Shift to move faster. The camera moves on a fixed 1/120 s step from real time (`bench/realtime/fly-camera.ts`), also while a frame is still rendering, so its speed does not depend on the frame rate.

**Classroom** uses the original render camera and evaluated geometry from Christophe Seux’s Blender Cycles Classroom. The web conversion maps the legacy material graphs to color textures and approximate PBR lobes, uses a warm directional sun, blue sky, ceiling emitters and a blackboard area emitter. The Cycles window sampling portals and the exterior fill-light proxy are omitted so the sky is visible and blinds can cast shadows. The point lamps, Cycles compositor, dust and volume passes are omitted. Most color textures are reduced to 512 px, with up to 1024 px for the blackboard. [Original Blender scene archive](https://download.blender.org/demo/test/classroom.zip)

**Sponza** uses Crytek’s atrium from the pinned Khronos glTF Sample Assets revision with the project’s atrium camera, sun and sky. The conversion preserves original base-color maps, glTF material factors and alpha masks at up to 512 px, and omits the normal and metallic-roughness maps. The model files, converted meshes and decoded textures stay in git-ignored asset directories. [Original Sponza and license](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Sponza)

### Lighting

**Render controls → Lighting** has the sun's strength, direction and size, and two switches: **Interior lights** (the ceiling lamps and the blackboard light) and **Sky light** (the sky seen through the windows; the sun is separate, set its strength to 0). The switches rewrite the light table and the emissive materials in the scene buffer, so nothing reloads and the cache relearns the new lighting while you watch. With everything off the image is black.

**Lamp brightness** scales the interior emitters (0 to 4 times what the scene ships with); the image responds linearly (denoised mean 0.044 with the lamps off, 0.083 at 0.5×, 0.122 at 1×, 0.200 at 2×).

### While the camera moves

Two settings change while the camera moves and return a quarter of a second after it stops: the cache trains fewer steps per frame (**Cache → Steps / frame while moving**, 1 instead of 2) and the upscaler keeps a shorter history (**Upscaler → Max history while moving**, 8 frames instead of 24: less trailing, more grain). Both are first guesses that have not been tuned by eye; set either to its still value to switch it off (`?stepsMoving=2&upHistoryMoving=24`).

### Starting warm, and slower GPUs

**Cache → Remember the trained network** (on) keeps the trained weights in the browser and loads them on the next visit, so the page does not start with plain path tracing behind the warm-up gate. A saved network is only used for the scene, settings and lighting it was trained under; **Reset cache** forgets it, `?saved=0` switches it off.

**Scene → Lower the resolution if the GPU is slow** (on) measures the first seconds at 960 × 540 and drops to 640 × 360 once if a frame takes longer than 50 ms; the network is kept across the switch. A size you pick yourself is remembered and ends the automatic choice (`?autoRes=0`, or any `?res=`).

### Bright specks while moving

The temporal upscaler runs with its **Auto pre-exposure** off here (fixed at 1, DIVERGES V344; reported as pmndrs/upscaler#118). With it on, a moving camera showed bright specks in the final stage only (values up to 139 where the denoised frame never exceeds 20; about 17 pixels per 1920 × 1080 frame, on the plain side as much as on the NRC side) and none when still. The raw and the denoised stages are the same moving or still. Turn the switch on under **Render controls → Upscaler**, or open the page with `?upAutoExposure=1`, to see the difference.

### Comparing reconstruction

**Accumulate while still** is on in the hero. The sample counter over each rendered arm shows the number of path samples averaged **before** filtering. Moving the camera, editing daylight, clearing the image, or changing views starts a new mean. Finishing the warm-up starts a new NRC mean. Network training continues while the camera is still. Turn accumulation off to see how reconstruction handles a fresh noisy frame; set both samples per frame to 1 for an equal-sample split. OIDN also has its own still-frame mean when progressive input is off.

**Render controls** exposes the path depth, per-arm sample budgets, optional luminance clamp, denoiser, SVGF weight, OIDN guides, upscaler path, sharpening and history. The firefly clamp operates on each traced sample before averaging and can dim real highlights. Zero disables it. Use **Bilinear** as the upscaler baseline and **Raw radiance** to inspect the sampler. **Display scale** at 1× lets you compare reconstruction at native resolution. Changing the resolution, depth, or training layout reloads the scene and restarts training.

Open **Classroom quality bench** from the hero, or run `node bench/live.ts nrc-classroom-quality`. **Run Classroom comparison** collects a plain 256-spp reference followed by five cold-start NRC runs at 640 × 360 with SVGF or OIDN and temporal upscaling to 1280 × 720. The table compares coherent 8 × 8 brightness changes and error to plain tracing for each stage. The displayed final captures use the same exposure as the interactive Classroom. The reference measures the web renderer’s own steady image. The supplied native screenshot is a visual target with different lighting, scene conversion, and reconstruction.

The quality page also links to a **native-scale comparison preset**. It traces one sample per frame at 1280 × 720 and presents at 1×, with accumulation enabled, path depth 10, a 102.8 luminance clamp and a 0.048° sun radius. Azimuth and elevation are rounded to 65° and 27° in our scene coordinates. This reproduces selected settings visible in the supplied control panel. Its renderer reported a 1920 × 1009 framebuffer on an RTX 4060 Laptop with NRD and TAA; our preset uses WebGPU, SVGF, and the temporal upscaler at 1×.

### Regular test bench

The scene IDs are `hero-classroom` and `hero-sponza`. Prepare either scene once with `python3 bench/hero/prepare.py classroom` or `python3 bench/hero/prepare.py sponza`. Both use the same scene descriptor, textures, daylight and camera as the hero.

| Entry point | What to tune |
| --- | --- |
| `node bench/live.ts nrc-pipeline` | Select **Classroom (color)** or **Sponza (color)** for the plain/NRC split, daylight, accumulation, denoiser and upscaler. |
| `node bench/live.ts nrc-live` | Select either interior for cache-only experiments with training, direction inputs, position bands and query depth. |
| `node bench/adapters/matrix.ts --scenes hero-classroom,hero-sponza --renderers pt-megakernel --spp 4` | Run the regular adapter comparison and export its report. |
| `nrc-pipeline-study` or `realtime-pipeline` in the bench runner | Set `scene=hero-classroom` for the pipeline study or `scenes=hero-classroom,hero-sponza` for the real-time pipeline. The NRC study renders its own reference for these interiors by default. |

For the interactive pages, `?scene=hero-classroom` or `?scene=hero-sponza` opens directly on that interior.

**SVGF input history (spp)** controls how much of the stationary trace feeds the filter. The interactive default is 16 samples. After the first 16 samples, each new sample gets a 1/16 weight. Zero selects the original unlimited mean. This affects SVGF's signal and material-factor averages together. The raw image and OIDN still use the full sample mean, and the sample badges count traced samples. On the quality page, select **Classroom (color)** or **Sponza (color)** and choose **Test stationary grain** to compare the two history settings over 1,024 stationary frames. [Measurements for both interiors](../results/NRC-CLASSROOM-QUALITY.md)

### GitHub Pages

**Deploy Classroom to GitHub Pages** builds a standalone Classroom presentation on relevant pushes to `main` or a manual workflow run. Set the repository’s **Settings → Pages → Source** to **GitHub Actions**. The deploy job reports the live URL. With the default project domain it is <https://pmndrs.github.io/pathtracer/>. The Pages jobs build the site from source and upload the generated scene, blue-noise texture, and OIDN weights as static files.

To reproduce the artifact locally after preparing Classroom, run `node bench/hero/build-pages.mjs`. The default base path is `/pathtracer/`. To serve the artifact at the origin root instead, run `PAGES_BASE=/ node bench/hero/build-pages.mjs` and `python3 -m http.server --directory dist/pages`. Classroom by Christophe Seux is listed under [CC0 on Blender’s demo files page](https://www.blender.org/download/demo-files/).
