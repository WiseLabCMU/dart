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
	$(DART) map -p results/$(TARGET)
	$(DART) slice -p results/$(TARGET)

video:
	$(DART) evaluate -p results/$(TARGET) -a -b $(BATCH)
	$(DART) evaluate -p results/$(TARGET) -ac -b $(BATCH)
	$(DART) video -p results/$(TARGET)

.phony: typecheck
typecheck:
	python -m mypy train.py
	python -m mypy manage.py
