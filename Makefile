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
