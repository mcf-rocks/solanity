OS := $(shell uname)

all:
ifeq ($(OS),Darwin)
SO=dylib
else
SO=so
all: cuda_crypt
endif

V=release

.PHONY:cuda_crypt
cuda_crypt:
	$(MAKE) V=$(V) -C src

DESTDIR ?= dist
install:
	mkdir -p $(DESTDIR)
ifneq ($(OS),Darwin)
	cp -f src/$(V)/libcuda-crypt.so $(DESTDIR)
endif
	ls -lh $(DESTDIR)

.PHONY:clean
clean:
	$(MAKE) V=$(V) -C src clean
