SHELL := /bin/bash
DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
docker-build:  ## Build the Docker image used in this project
	pushd . ; cd scripts/dockerf/ ; bash setup-project.sh; popd
env-notebook:  ## Start a Jupyter notebook
	docker-compose \
	-p mlops \
	-f scripts/dockerf/docker-compose.jupyter.yml \
	up
linter:  ## Lint churn_library file
	docker-compose \
	-p mlops \
	-f scripts/dockerf/docker-compose.py.yml \
	run --rm -w /mlops-predict-churn-clean-code mlops \
	bash scripts/linter-code.sh churn_library.py
bash:  ## Open an interactive terminal in Docker container
	docker-compose \
	-p mlops \
	-f scripts/dockerf/docker-compose.py.yml \
	run --rm -w /mlops-predict-churn-clean-code mlops
tests:  ## Test all functions implemented in churn_library
	docker-compose \
	-p mlops \
	-f scripts/dockerf/docker-compose.py.yml \
	run --rm -w /mlops-predict-churn-clean-code mlops \
		bash  scripts/test-codes.sh logs/*.log churn_script_logging_and_tests.py
