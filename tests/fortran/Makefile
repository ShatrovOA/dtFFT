#!/usr/bin/make

include ../../Makefile.inc

SRCS = $(wildcard *.F90)
EXES = $(addprefix ../../$(EXE_DIR)/,$(basename $(SRCS)))

.PHONY : all
all: $(EXES)

../../$(EXE_DIR)/%: %.F90 ../../$(OBJ_DIR)/dtfft.o
	@echo Building $<
	@$(FC) -o $@ $< $(FC_LINK_FLAGS)
	@mv $(basename $<).gcno ../../$(OBJ_DIR) 2>/dev/null; true

.PHONY: clean
clean:
	@rm -rf $(EXES)

comm_size? = 0
check: all
	@for test in $(EXES) ; do \
			np=$$(( $${np:-$(comm_size)}+1 )); \
			mpiexec --oversubscribe -np $$np ./$$test ; \
	done
	@mv *.gcda ../../$(OBJ_DIR)/ 2>/dev/null; true
