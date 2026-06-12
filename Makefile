SHELL := /bin/bash
.DEFAULT_GOAL := help

PROJECT_ROOT := $(CURDIR)
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
ifeq ($(UNAME_M),x86_64)
LOCAL_PLATFORM := linux-x86_64
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
LOCAL_PLATFORM := linux-arm64
endif
else ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),x86_64)
LOCAL_PLATFORM := macos-x86_64
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
LOCAL_PLATFORM := macos-arm64
endif
endif

.PHONY: help bundle sync-test-env test test-integration test-bundle test-full coverage publish

help: # @help Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?# @help ' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?# @help "}; {printf "  %-18s %s\n", $$1, $$2}'

bundle: # @help Copy local-platform bundled executables from sister worktrees into src/trillim/_bin
ifeq ($(LOCAL_PLATFORM),)
	@echo "Unsupported local platform: $(UNAME_S) $(UNAME_M)" >&2; exit 1
else
	uv run python -c "from scripts import build_wheels; build_wheels.copy_local_binaries('$(LOCAL_PLATFORM)')"
endif

sync-test-env:
	uv sync --extra dev --extra voice

test: sync-test-env # @help Sync dev/voice extras and run the fast deterministic unittest suite
	uv run python -m unittest discover -q

test-integration: sync-test-env # @help Run unittest suite with real runtime integration tests enabled
	TRILLIM_RUN_INTEGRATION=1 uv run python -m unittest discover -q

test-bundle: sync-test-env bundle # @help Copy local binaries and run bundled-binary package smoke tests
	TRILLIM_RUN_BUNDLE_TESTS=1 uv run python -m unittest tests.quantize.test_config_manifest.QuantizeBundleTests -q

test-full: test-integration test-bundle # @help Run integration tests and bundled-binary smoke tests

coverage: sync-test-env # @help Sync dev/voice extras and print the terminal coverage report
	COVERAGE_PROCESS_START="$(PROJECT_ROOT)/pyproject.toml" uv run --extra dev python -m coverage erase
	COVERAGE_PROCESS_START="$(PROJECT_ROOT)/pyproject.toml" uv run --extra dev python -m coverage run -m unittest discover -q
	uv run --extra dev python -m coverage combine
	uv run --extra dev python -m coverage report -m

publish: # @help publish packaged wheels to PyPI
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Worktree must be clean before publish" >&2; \
		exit 1; \
	fi
	@if ! case "$(BUMP)" in major|minor|patch) true ;; *) false ;; esac; then \
		echo "BUMP must be one of: major, minor, patch" >&2; \
		exit 1; \
	fi
	rm -rf dist/*
	uv version --bump $(BUMP)
	uv run scripts/build_wheels.py
	source .env && uv publish dist/* --token $$UV_PUBLISH_TOKEN
	git add pyproject.toml uv.lock
	git commit -m "$(BUMP) project version bump"
