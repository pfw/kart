PYTHON_PDAL_REF ?= 3.1.2
PYTHON_PDAL_REPO ?= PDAL/python
PYTHON_PDAL_ARCHIVE := python-$(PYTHON_PDAL_REF).tar.gz

SHELL = /bin/bash
export PREFIX ?= $(abspath env)

ifeq ($(OS),Windows_NT)
	PLATFORM := Windows
else
	PLATFORM := $(shell uname -s)
endif

ifeq ($(PLATFORM),Darwin)
	CCACHE_PATH = /usr/local/opt/ccache/bin
	LIBSUFFIX = dylib
	PY3 ?= $(realpath /Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7)
	WHEELTOOL = delocate
	export LDFLAGS+=-Wl,-rpath,'@loader_path/../../../'
else ifeq ($(PLATFORM),Linux)
	LIBSUFFIX = so
	CCACHE_PATH = /usr/lib/ccache
	CCACHE_PATH := $(or $(CCACHE_PATH),/usr/lib64/ccache)
	WHEELTOOL = auditwheel
	export LDFLAGS := -Wl,-rpath='$$ORIGIN/../../..'
endif
PY3 := $(or $(PY3),python3.7)

# use ccache if available
export PATH := $(CCACHE_PATH):$(PREFIX)/bin:$(PATH)

# default target
.PHONY: all
all: wheel

.PHONY: clean
clean:
	-$(RM) -r env
	-$(RM) -r wheel wheelhouse
	-$(RM) -r src/build src/dist src/.eggs

.PHONY: cleaner
cleaner: clean
	-(cd src && python3 setup.py clean)

.PHONY: cleanest
cleanest: clean
	-$(RM) -r src

.PHONY: clean-configure
clean-configure:

#
# Download Archives
#

$(PYTHON_PDAL_ARCHIVE):
	wget https://github.com/$(PYTHON_PDAL_REPO)/archive/$(PYTHON_PDAL_REF).tar.gz -O $@

.PHONY: archive
archive: $(PYTHON_PDAL_ARCHIVE)

#
# Extract Archives
#

src: $(PYTHON_PDAL_ARCHIVE)
	rm -rf $@
	mkdir -p $@
	@echo "Extracting $(PYTHON_PDAL_ARCHIVE) ..."
	tar xzf $(PYTHON_PDAL_ARCHIVE) --strip-components=1 -C $@


.PHONY: source
source: src

#
# PDAL
#

.PHONY: install
install: | src $(libgit2)
	@echo 'LDFLAGS=$(LDFLAGS) PY3=$(PY3)'
	pip3 install -e src

.PHONY: wheel
wheel: | src $(libgit2)
	@echo 'LDFLAGS=$(LDFLAGS) PY3=$(PY3)'
	cd src && python3 setup.py  --no-user-cfg bdist_wheel -d $(abspath $@)