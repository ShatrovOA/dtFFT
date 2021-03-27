#!/usr/bin/make

include ../Makefile.inc

DOBJ = ../$(OBJ_DIR)

.PHONY: lib
lib: $(DOBJ)dtfft.o

$(DOBJ)dtfft.o: dtfft.F90 \
	$(DOBJ)dtfft_plan_c2c_m.o \
	$(DOBJ)dtfft_plan_r2r_m.o \
	$(DOBJ)dtfft_plan_r2c_m.o \
	$(DOBJ)dtfft_c_interface.o \
	$(DOBJ)dtfft_c_interface_c.o \
	$(DOBJ)dtfft_parameters.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_plan_c2c_m.o: dtfft_plan_c2c_m.F90 \
	$(DOBJ)dtfft_plan_base_m.o \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_abstract_executor_m.o \
	$(DOBJ)dtfft_executor_fftw_m.o \
	$(DOBJ)dtfft_executor_mkl_m.o \
	$(DOBJ)dtfft_executor_cufft_m.o \
	$(DOBJ)dtfft_info_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_plan_r2c_m.o: dtfft_plan_r2c_m.F90 \
	$(DOBJ)dtfft_plan_base_m.o \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_abstract_executor_m.o \
	$(DOBJ)dtfft_executor_fftw_m.o \
	$(DOBJ)dtfft_executor_mkl_m.o \
	$(DOBJ)dtfft_executor_cufft_m.o \
	$(DOBJ)dtfft_plan_c2c_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_info_m.o: dtfft_info_m.F90 \
	$(DOBJ)dtfft_precisions.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_abstract_executor_m.o: dtfft_abstract_executor_m.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_info_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 
 
$(DOBJ)dtfft_plan_r2r_m.o: dtfft_plan_r2r_m.F90 \
	$(DOBJ)dtfft_plan_base_m.o \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_abstract_executor_m.o \
	$(DOBJ)dtfft_executor_fftw_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_interface_fftw_m.o: dtfft_interface_fftw_m.F90 \
	$(DOBJ)dtfft_precisions.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) $(FFTW3_INCLUDE) -c $< -o $@ 

$(DOBJ)dtfft_executor_fftw_m.o: dtfft_executor_fftw_m.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_info_m.o \
	$(DOBJ)dtfft_abstract_executor_m.o \
	$(DOBJ)dtfft_interface_fftw_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_interface_mkl_m.o: dtfft_interface_mkl_m.F90 \
	$(DOBJ)dtfft_precisions.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) $(MKL_INCLUDE) -c $< -o $@ 

$(DOBJ)dtfft_executor_mkl_m.o: dtfft_executor_mkl_m.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_info_m.o \
	$(DOBJ)dtfft_abstract_executor_m.o \
	$(DOBJ)dtfft_interface_mkl_m.o \
	$(DOBJ)dtfft_parameters.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_executor_cufft_m.o: dtfft_executor_cufft_m.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_info_m.o \
	$(DOBJ)dtfft_abstract_executor_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_parameters.o: dtfft_parameters.F90 \
	$(DOBJ)dtfft_precisions.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_plan_base_m.o: dtfft_plan_base_m.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_transpose_m.o \
	$(DOBJ)dtfft_parameters.o \
	$(DOBJ)dtfft_info_m.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_transpose_m.o: dtfft_transpose_m.F90 \
	$(DOBJ)dtfft_info_m.o \
	$(DOBJ)dtfft_precisions.o
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_precisions.o: dtfft_precisions.F90
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

$(DOBJ)dtfft_c_interface_c.o: dtfft_c_interface_c.c
	@echo $(COTEXT)
	@$(CC) $(CC_BUILD_FLAGS) -c $< -o $@

$(DOBJ)dtfft_c_interface.o: dtfft_c_interface.F90 \
	$(DOBJ)dtfft_precisions.o \
	$(DOBJ)dtfft_plan_r2r_m.o \
	$(DOBJ)dtfft_plan_c2c_m.o \
	$(DOBJ)dtfft_plan_r2c_m.o 
	@echo $(COTEXT)
	@$(FC) $(FC_BUILD_FLAGS) -c $< -o $@ 

.PHONY: coverage
coverage:
	@gcov -o ../$(OBJ_DIR) *
	@mv *.gcov ../