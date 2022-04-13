PDAL_REF ?= 2.4.0
PDAL_REPO ?= PDAL/PDAL
PDAL_ARCHIVE := PDAL-$(PDAL_REF).tar.gz

CFLAGS += -g
CXXFLAGS += -g

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
	CMAKE_PARAMS = -DCMAKE_INSTALL_NAME_DIR="@rpath"
else ifeq ($(PLATFORM),Linux)
	CCACHE_PATH = /usr/lib/ccache
	CCACHE_PATH := $(or $(CCACHE_PATH),/usr/lib64/ccache)
	LIBSUFFIX = so
	CMAKE_PARAMS = -DCMAKE_INSTALL_NAME_DIR="$$ORIGIN"
endif

# use ccache if available
export PATH := $(CCACHE_PATH):$(PREFIX)/bin:$(PATH)

configure-pdal = build/Makefile
build-pdal = build/libpdalcpp.$(LIBSUFFIX)

# default target
.PHONY: all
all: $(build-pdal)

.PHONY: clean
clean:
	-$(RM) -r env
	-$(RM) -r build/pdal.*

.PHONY: cleaner
cleaner: clean
	-$(MAKE) -C build/ clean

.PHONY: cleanest
cleanest:
	-$(RM) -r build/ src/

.PHONY: clean-configure
clean-configure:
	-$(RM) $(configure-pdal)

#
# Download Archives
#

$(PDAL_ARCHIVE):
	wget https://github.com/$(PDAL_REPO)/archive/$(PDAL_REF).tar.gz -O $@

.PHONY: archive
archive: $(PDAL_ARCHIVE)

#
# Extract Archives
#

src: $(PDAL_ARCHIVE)
	rm -rf $@
	mkdir -p $@
	@echo "Extracting $(PDAL_ARCHIVE) ..."
	tar xzf $(PDAL_ARCHIVE) --strip-components=1 -C $@


.PHONY: source
source: src

$(configure-pdal): export PKG_CONFIG_PATH=/dev/null
$(configure-pdal): export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(PREFIX)/lib
$(configure-pdal): | src
	cmake $(CMAKE_PARAMS) \
		-DCMAKE_POLICY_DEFAULT_CMP0074=NEW \
		-DCMAKE_PREFIX_PATH=$(PREFIX) \
		-DCMAKE_LIBRARY_PATH=$(PREFIX)/lib \
		-DCMAKE_INCLUDE_PATH=$(PREFIX)/include \
		-DGDAL_ROOT=$(PREFIX) \
		-DGDAL_LIBRARY=$(PREFIX)/lib/libgdal.dylib \
		-DGDAL_INCLUDE_DIR=$(PREFIX)/include \
		-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
		-DCMAKE_INSTALL_RPATH=$(PREFIX)/lib \
		-S src -B build

.PHONY: configure
configure: clean-configure $(configure-pdal)

$(build-pdal): $(configure-pdal)
	cmake \
		--build build \
		--config RelWithDebInfo
	$(MAKE) print-lib-deps-$(PLATFORM)
	# $(MAKE) check-lib-deps-$(PLATFORM)

#
# INSTALLATION
#

.PHONY: install
install: $(build-pdal)
	cmake \
		--install build \
		--config RelWithDebInfo \
		--prefix $(PREFIX)

.PHONY: print-lib-deps
print-lib-deps: print-lib-deps-$(PLATFORM)

.PHONY: print-lib-deps-Darwin
print-lib-deps-Darwin:
	@echo -e "\nExamining library dependencies..."
	cd build/ && find . \( -name "*.dylib" -o -name "*.so" \) -exec otool -L {} \;

.PHONY: print-lib-deps-Linux
print-lib-deps-Linux:
	@echo -e "\nExamining library dependencies..."
	cd build/  && find . -name "*.so" -print -exec ldd {} \;

.PHONY: check-lib-deps
check-lib-deps: check-lib-deps-$(PLATFORM)

.PHONY: check-lib-deps-Darwin
check-lib-deps-Darwin:
	@echo -e "\nLooking for missing library dependencies..."
	find build \( -name "*.dylib" -o -name "*.so" \) | xargs -t -I % sh -c '! (otool -l % | grep /usr/local/ )'

.PHONY: check-lib-deps-Linux
check-lib-deps-Linux:
	@echo -e "\nLooking for missing library dependencies..."
	find build -name "*.so" | xargs -t -I % sh -c '! ( ldd % | grep -P " => /usr/(local|lib)/(?!x86_64-linux-gnu)" )'