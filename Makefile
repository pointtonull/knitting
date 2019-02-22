SRC = src
PYTHON = python3
REQUIREMENTS = requirements.txt


.PHONY: clean
clean:
	@echo "Cleaning all artifacts..."
	@-rm .deps
	@-rm -rf $(DEPS)


deps: .deps

.deps: $(REQUIREMENTS)
	pip install -qUr $(REQUIREMENTS)
	touch .deps


.PHONY: tdd
tdd: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v --exitfirst --pdb ../tests


.PHONY: unit test
unit test: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v --doctest-modules ../tests ./


.PHONY: coverage
coverage: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v ./ --cov=./ --cov-report=term-missing ../tests

.PHONY: ipython
ipython: deps
	cd $(SRC);\
	DUMMY=True PYTHONPATH=deps $(PYTHON) terminal.py

.PHONY: run
run: deps
	cd $(SRC);\
	DUMMY=True PYTHONPATH=deps $(PYTHON) main.py

