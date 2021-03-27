#!/usr/bin/make

PHONY: tests
tests: f_tests c_tests

c_tests:
	@cd c; $(MAKE)

f_tests:
	@cd fortran; $(MAKE)

check:
	@cd fortran; $(MAKE) $@
	@cd c;			 $(MAKE) $@