set(TEST_SOURCES
	test_backend_mpi.F90
)
if ( DTFFT_WITH_COMPRESSION )
	list(APPEND TEST_SOURCES "test_backend_compressed.F90")
endif()

foreach(test ${TEST_SOURCES})
	get_filename_component(test_name ${test} NAME_WLE)
	# Backend tests are all about exchanging data between ranks. With a single rank there is
	# nothing to exchange: every backend degenerates into a self copy, which also bypasses
	# compression completely, so tests start from 2 ranks.
	# Run with 2..DTFFT_TEST_MAX_PROCS ranks so both non-zero and zero-count paths are covered.
	# With comm_size > NY_ROWS (=7) some ranks get zero-sized slabs automatically.
	dtfft_add_test(${test_name} ${test}
		TEST_UTILS
		WITH_RMA
		LINKER_LANGUAGE Fortran
		MIN_PROCS 2
		NAME_NPROCS
		SKIP_RETURN_CODE 77
	)
endforeach()