SRC = src
PYTHON = python


.PHONY: clean
clean:
	@echo "Cleaning all artifacts..."
	@-rm .deps
	@-rm -rf $(DEPS)


deps: .deps
.deps: $(REQUIREMENTS) requirements.txt
	pip install -qur requirements.txt
	touch .deps


ipython: deps
	cd $(SRC);\
	$(PYTHON) -m IPython


.PHONY: tdd
tdd: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v --exitfirst --docttest-modules --pdb ../tests


.PHONY: unit test
unit test: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v --doctest-modules ../tests ./


.PHONY: coverage
coverage: deps
	cd $(SRC);\
	$(PYTHON) -m pytest -v --doctest-modules ./ --cov=./ --cov-report=term-missing ../tests


.PHONY: debug
debug: deps
	# TODO: reimplement using pytests
	cd $(SRC);\
	$(IPYTHON) --pdb -m main
