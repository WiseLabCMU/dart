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

Cabinets:
```sh
python train.py -s data/cabinets/cabinets.json -o results/cabinets-sim -e 2 -p data/cabinets/sim.mat --weight sqrt --min_speed 0.25
python evaluate.py -p results/cabinets-sim; python examples.py results/cabinets-sim; python map.py -p results/cabinets-sim
```

Motion Stage:
```sh
python train.py -s data/linear1/linear1.json -o results/linear1.b -e 2 -p data/linear1/linear1.mat --clip 99.99 --min_speed 0.005 -b 512 -e 50 --weight sqrt
python evaluate.py -p results/linear1.b; python examples.py -p results/linear1.b; python map.py -p results/linear1.b
```
