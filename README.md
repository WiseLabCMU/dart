# Doppler-Aided Radar Tomography using Neural Reflectance Fields

## Dependencies

```sh
sudo apt-get install libhdf5-dev
```

Install JAX GPU, then requirements.txt.

## Params

Max batch size:
- cup, simulation: 2048
- cabinets, cabinets2: 512

## Commands

```
CUDA_VISIBLE_DEVICES=1 python simulate.py -s data/cabinets/cabinets-180.json -o data/cabinets/simulated.mat -g data/cabinets/map.mat -j data/cabinets/cabinets.mat
CUDA_VISIBLE_DEVICES=1 python train.py -s data/cabinets/cabinets-180.json -p data/cabinets/simulated.mat -e 5 -o results/cabinets_sim
CUDA_VISIBLE_DEVICES=1 python evaluate.py -p results/cabinets_sim
```
