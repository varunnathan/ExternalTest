# Default target
.PHONY: all
all: help

# Help target to display available commands
.PHONY: help
help:
    @echo "Usage:"
    @echo "  make train     - Run run.sh with MODE=train"
    @echo "  make test      - Run run.sh with MODE=test"
    @echo "  make validate  - Run run.sh with MODE=valid"

# Target to run run.sh with MODE=train
.PHONY: train
train:
    bash run.sh -m train

# Target to run run.sh with MODE=test
.PHONY: test
test:
    bash run.sh -m test

# Target to run run.sh with MODE=validate
.PHONY: validate
validate:
    bash run.sh -m valid