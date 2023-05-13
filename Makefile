ifndef RADIUS
RADIUS=0.6
endif

ifndef BATCH
BATCH=32
endif

ifndef DART
DART=python manage.py
endif

.phony: eval typecheck eval-all video

eval:
	$(DART) evaluate -p $(TARGET) -b $(BATCH)
	$(DART) examples -p $(TARGET)
	$(DART) map -p $(TARGET) -r $(RADIUS)

video:
	$(DART) evaluate -p $(TARGET) -a
	$(DART) evaluate -p $(TARGET) -ac

typecheck:
	python -m mypy train.py
	python -m mypy manage.py
