ifndef BATCH
BATCH=2
endif
ifndef DART
DART=python manage.py
endif
ifndef TRAIN
TRAIN=python train.py
endif
ifndef METHOD
METHOD=ngpsh
endif

.phony: visualize experiment
experiment: train visualize
visualize: slices video

.phony: train slices video
train:
	$(TRAIN) $(METHOD) -p data/$(DATASET) -o results/$(TARGET) $(FLAGS)

slices:
	$(DART) map -p results/$(TARGET) --resolution 50
	$(DART) slice -p results/$(TARGET)

video:
	$(DART) evaluate -p results/$(TARGET) -a -b $(BATCH)
	$(DART) evaluate -p results/$(TARGET) -ac -b $(BATCH)
	$(DART) ssim -p results/$(TARGET)
	$(DART) video -p results/$(TARGET)

.phony: typecheck
typecheck:
	python -m mypy train.py
	python -m mypy manage.py

.phony: baselines
baselines:
	$(DART) gt_map -p data/$(DATASET) $(FLAGS)
	$(DART) simulate -p data/$(DATASET)
	$(DART) ssim --baseline -p results/$(TARGET)
	$(DART) ssim_synthetic -p results/$(TARGET)
