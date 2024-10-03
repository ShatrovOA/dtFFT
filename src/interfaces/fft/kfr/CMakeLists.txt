find_package(KFR REQUIRED)

target_link_libraries(dtfft kfr_capi kfr)

get_target_property(kfr_imported_options kfr INTERFACE_COMPILE_OPTIONS)
list(REMOVE_ITEM kfr_imported_options "-faligned-allocation")
set_target_properties(kfr PROPERTIES INTERFACE_COMPILE_OPTIONS "${kfr_imported_options}")

target_sources(dtfft PRIVATE
  dtfft_executor_kfr_m.F90
  dtfft_interface_kfr_m.F90
  dtfft_interface_kfr.c
)
