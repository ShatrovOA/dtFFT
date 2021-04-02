#!/usr/bin/make

.DEFAULT_GOAL := lib

include Makefile.inc

MKDIRS  = $(OBJ_DIR) $(MOD_DIR) $(EXE_DIR)

.PHONY : all
all: lib tests

.PHONY: lib
lib: $(MKDIRS)
	@cd src; $(MAKE) $@
	@echo $(LITEXT)
	@$(STATIC_LIB_COMMAND)
	@$(SHARED_LIB_COMMAND)

check: tests
	@cd tests; $(MAKE) $@

.PHONY : tests
tests: lib
	@cd $@ ; $(MAKE) $@

.PHONY: coverage
coverage: check
	@cd src; $(MAKE) $@

.PHONY : $(MKDIRS)
$(MKDIRS):
	@mkdir -p $@

.PHONY: doc
doc:
	@ford doc/dtfft_ford.md

.PHONY : clean
clean:
	@echo Deleting $(BUILD_DIR) directory
	@rm -fr $(BUILD_DIR)
	@echo Deleting $(EXE_DIR) directory
	@rm -fr $(EXE_DIR)
	@rm -fr *.gcov
	@cd src; $(MAKE) $@