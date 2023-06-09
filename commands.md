## Experiments

cichall-sh2: baseline
cichall-sh3: pow(10, db/20)
cichall-sh4: pow(10, db/10)

### cichall

```sh
python train.py ngp -p data/cichall -o results/cichall -e 10 -b 1024 --iid
TARGET=results/cichall LOWER="-7 -15 -2" UPPER="13 9 3" RESOLUTION="800 1000 200" make visualize
```

### cabinets

```sh
python train.py ngp -p data/cabinets-10k -o results/cabinets-10k -e 5 --iid -b 512
TARGET=results/cabinets-10k LOWER="-5 -5 -1" UPPER="7 7 3" RESOLUTION="600 600 200" make visualize
```

### aframe

```sh
python train.py ngp -p data/aframe_t265 -o results/aframe -e 10
TARGET=results/aframe LOWER="-8 -6 -1" UPPER="4 6 3" RESOLUTION="600 600 200" make visualize
```

### picnic

```sh
python train.py ngp -p data/picnic-2 -o results/picnic-2 -b 1024 --iid -e 3
TARGET=results/picnic-2 LOWER="-8 -18 -1" UPPER="6 -6 3" RESOLUTION="700 600 200" make vizualize
```


***

### Cabinets

Simulate/Train:
```sh
python manage.py simulate -s data/cabinets/cabinets.json -o data/cabinets-000/sim.mat -g data/cabinets/map.mat -j data/cabinets-000/cabinets-000.mat
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets.sim -e 5 --repeat 5 -p data/cabinets-000/sim.mat --min_speed 0.25 --iid
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets.real -p data/cabinets-000/cabinets-000.mat --norm 1e4 --min_speed 0.25 -e 5 --repeat 5 --iid
python train.py ngpsh -s data/cabinets/cabinets.json -o results/cabinets.ngpsh -p data/cabinets-000/cabinets-000.mat --norm 1e4 --min_speed 0.25 -e 5 --repeat 5 --iid
```

Create video:
```sh
TARGET=results/cabinets.sim make video
TARGET=results/cabinets.real make video
TARGET=results/cabinets.ngpsh make video
python manage.py video -p results/cabinets.sim results/cabinets.real -f 30 -s 512 -o results/cabinets.mp4
```

### Cabinets (5 traces)

Simulate/Train:
```sh
python manage.py simulate -s data/cabinets/cabinets.json -o data/cabinets-5/sim.mat -g data/cabinets/map.mat -j data/cabinets-5/cabinets-5.mat
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets-5.sim -e 10 --repeat 1 -p data/cabinets-5/sim.mat --min_speed 0.25 --iid
python train.py ngp -s data/cabinets/cabinets.json -o results/cabinets-5.real -p data/cabinets-5/cabinets-5.mat --norm 1e4 --min_speed 0.25 -e 5 --repeat 1 --iid
```

### Couch

```sh
python train.py ngp -s data/couch0/dataset0.json -o results/couch0 -p data/couch0/dataset0.mat --min_speed 0.25 -b 2048 -e 5 --repeat 5 --iid --norm 1e5
TARGET=results/couch0 make video
python manage.py video -p results/couch0 -f 30 -s 512 -o results/couch0.mp4
```

### Motion Stage

Simulate:
```sh
python manage.py simulate -s data/linear1/linear1.json -o data/linear1/sim.mat -g data/linear1/map.mat -j data/linear1/linear1.mat
```

Train:
```sh
python train.py ngp -s data/linear1/linear1-fixed.json -o results/linear1.sim -p data/linear1/sim.mat --min_speed 0.005 -b 512 -e 5 --repeat 10 --iid
python train.py ngp -s data/linear1/linear1-fixed.json -o results/linear1.real -p data/linear1/linear1.mat --norm 1e6 --min_speed 0.005 -b 512 -e 5 --repeat 10 --iid
```

Create video:
```sh
TARGET=results/linear1.sim make video
TARGET=results/linear1.real make video
python manage.py video -p results/linear1.sim results/linear1.real -f 15 -s 512 -o results/linear1.mp4
```
