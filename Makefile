ifndef BATCH
BATCH=4
endif
ifndef DART
DART=python manage.py
endif

.phony: eval typecheck video slices visualize

visualize: video slices

video:
	$(DART) evaluate -p $(TARGET) -a -b $(BATCH)
	$(DART) evaluate -p $(TARGET) -ac -b $(BATCH)
	$(DART) video -p $(TARGET)

slices:
	$(DART) map -p $(TARGET)
	$(DART) slice -p $(TARGET)

typecheck:
	python -m mypy train.py
	python -m mypy manage.py
