# Datasets

## DART Datasets

Our datasets can be downloaded from [Zenodo](https://zenodo.org/records/10938617); the total size is 42.1GB (compressed).

**NOTE**: our datasets have been migrated from Google Drive to Zenodo due to recent storage limit changes. Each trace has been compressed into a `.zip` archive to fit in Zenodo's 50GB limit, and will need to be decompressed after downloading.

These datasets were collected using our data collection platform, [Rover](https://github.com/thetianshuhuang/rover), which creates `radar.h5` and `trajectory.h5` data files, along with `sensor.json` and `metadata.json` dataset information files.

| Dataset      | Description |
|------------- | ----------- |
| apartment-1  | High-rise apartment (kitchen/living room only)
| apartment-2  | High-rise apartment (kitchen/living room, bedroom) |
| house-1      | Early 20th century house (ground floor) |
| house-2      | Early 20th century house (ground & 2nd floor) |
| lab-1        | 5 boxes of varying material in a lab space |
| lab-2        | 5 boxes of varying material in a lab space |
| office-1     | Office space |
| office-2     | Office Space |
| rowhouse-1   | Townhouse apartment (kitchen/living room, 1 bedroom) |
| rowhouse-2   | Townhouse apartment (kitchen/living room, hallway) |
| rowhouse-3   | Townhouse apartment (kitchen/living room, bathroom, 2 bedrooms) |
| yard         | Backyard and garage |

## Training Dataset Format

DART expects the following arrays, where axis 0 in each array represents a data point, which consists of a doppler column:

| Key, Type | Description |
| --- | ----------- |
| `x : Float[n, 3]` | Radar position (4); axis 0/1 are horizontal, and axis 2 is vertical. |
| `A : Float[n, 3, 3]` | Radar rotation (1) matrix (radar orientation -> front-left-up) |
| `v : Float[n, 3]` | Radar velocity direction vector (2) |
| `s : Float[n]` | Radar speed (2,4) |
| `p, q: Float[n, 3]` | Vectors which form an orthonormal basis along with `v` |
| `weight: Float[n]` | Integration weight of each sample in this doppler column (3) |
| `doppler: Float[n]` | Doppler velocity (4) of this column |
| `doppler_idx: Int[n]` | Local index of this particular doppler velocity (metadata only) |
| `frame_idx: Int[n]` | Global index of this particular frame (metadata only) |
| `i: Int[n]` | Global index of this doppler column (metadata only) |
| `rad: Float[n, range, azimuth]` | Actual values of this doppler column (4) |

Notes:
1. Our coordinate convention is FLU (Front-Left-Up), where the sensor FOV is centered around +x, +y is to the left of +x, and +z is straight up.
2. `v` is a normal vector, describes the velocity along with `s`, and forms an orthonormal basis along with `p, q`.
3. The integration weight captures (and pre-computes) the impact of arc length and speed on numerical integration.
4. Units are in m, m/s where applicable. The actual radar bin value does not have a specific unit due to the difficulty of propagating interpretable scaling magnitudes through the entire radar collection and processing pipeline.

These values also include some redundant precomputed information. If working with the output from our data collection platform, [Rover](https://github.com/thetianshuhuang/rover) (`radar.h5` and `trajectory.h5`), `data.h5` can be created using `dataset.py`:
```sh
python manage.py dataset -p <path/to/dataset> --val <0.2>
```