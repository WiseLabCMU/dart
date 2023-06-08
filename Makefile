ifndef RADIUS
RADIUS=0.6
endif
ifndef BATCH
BATCH=32
endif
ifndef DART
DART=python manage.py
endif
ifndef LOWER
LOWER=-3 -3 -3
endif
ifndef UPPER
UPPER=3 3 3
endif
ifndef RESOLUTION
RESOLUTION=400 400 400
endif


.phony: eval typecheck video slices visualize

eval:
	$(DART) evaluate -p $(TARGET) -b $(BATCH)
	$(DART) examples -p $(TARGET)

visualize: video slices

video:
	$(DART) evaluate -p $(TARGET) -a -b 1
	$(DART) evaluate -p $(TARGET) -ac -b 1
	$(DART) video -p $(TARGET)

slices:
	$(DART) map -p $(TARGET) -l $(LOWER) -u $(UPPER) -r $(RESOLUTION) -b 16
	$(DART) slice -p $(TARGET)

typecheck:
	python -m mypy train.py
	python -m mypy manage.py
