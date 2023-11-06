#  ____________       ______  _______
#     _______  \  /\ (_____ \(_______)
#      ___   \  \/  \ _____) )_
#   _____ |   | / /\ (_____ (| |
#     ___ |__/ / /  \ \    | | |___
#   __________/_/    \_|   |_|\____)
#     Doppler Aided Radar Tomography
#

BATCH?=2
DART?=python manage.py
TRAIN?=python train.py
METHOD?=ngpsh

.phony: experiment
experiment: train slices evaluate

.phony: train slices evaluate video
train:
	$(TRAIN) $(METHOD) -p data/$(DATASET) -o results/$(TARGET) $(FLAGS)

slices:
	$(DART) map -p results/$(TARGET) --resolution 50
	$(DART) slice -p results/$(TARGET)

evaluate:
	$(DART) evaluate -p results/$(TARGET) -a -b $(BATCH)
	$(DART) ssim -p results/$(TARGET)

video: evaluate
	$(DART) evaluate -p results/$(TARGET) -ac -b $(BATCH)
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
	$(DART) ssim --psnr 35 -p results/$(TARGET)
	$(DART) ssim --psnr 30 -p results/$(TARGET)
	$(DART) ssim --psnr 25 -p results/$(TARGET)
