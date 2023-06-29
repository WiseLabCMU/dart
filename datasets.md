# Datasets

## Summary

| Dataset | Paper Name | Size | Time Offset | Mapping Layer |
| --- | --- | --- | --- | --- |
| arena | Lab | 2.8GB | 0.0 | 105 |
| arena-plus | Lab+ | 5.9GB | 0.3 | 108 |
| cichallway | Hall A | 3.4GB | 0.0 | 106 |
| ciccorridor | Hall B | 3.0GB | 0.6 | 120 |
| cafe | Cafe | 4.3GB | -0.3 | |
| arena-plus-plus | Lab++ | 5.4GB | 0.8 | - |
| prius | Car | 5.1GB | 0.7 | - |
| tent | Tent | 3.8GB | -0.4 | - |

## Novel View Synthesis

### arena

Walking around the ARENA. Time sync offset: 0.0.

```
--lower -7.160709619522095 -8.436034202575684 -2.531515896320343 --upper 5.535109639167786 4.2077077478170395 2.529978632926941
```

### arena-plus

Walking around ARENA and just outside (couch area, conference room). Time sync offset: 0.3.

```
--lower -8.499373435974121 -14.147072792053223 -2.027274349704385 --upper 5.184354662895203 4.13895808160305 2.9735187292099
```

### cichallway

Walking in the hallway outside of CIC . Time sync offset: 0.0.

```
--lower -8.076507091522217 -16.01803970336914 -2.0800934955477715 --upper 12.32780647277832 14.856937408447266 2.412882834672928
```

### ciccorridor

Same as cichallway. Time sync offset: 0.6
```
--lower -7.7983503341674805 -11.238823413848877 -2.017005732282996 --upper 17.145482063293457 16.85133647918701 2.6927783489227295
```

### cafe

A cafe space. Time sync offset: -0.6.
```
--lower -10.659505844116211 -5.171474933624268 -2.5155683159828186 --upper 7.666897535324097 10.089675903320312 2.772848069667816
```

## Mapping

### arena-plus-plus (2D)

Entire CIC 2300 area. Time sync offset: 0.8.

## Imaging

Resolution = 100
Trained with IID val

### prius (3D)

Prius scanned from all angles. Time sync offset: 0.7.

NOTE: override on bounds.
```
--lower -3 -5 -3 --upper 3 5 2
```

### tent (3D)

Tent with a cabinet hidden inside. Time sync offset: -0.4.

NOTE: override on bounds.
```
--lower -4 -9 -2 --upper 0 -4 1
```
