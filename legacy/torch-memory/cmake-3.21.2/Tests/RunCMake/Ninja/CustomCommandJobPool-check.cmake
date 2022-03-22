set(log "${RunCMake_BINARY_DIR}/CustomCommandJobPool-build/build.ninja")
file(READ "${log}" build_file)
if(NOT "${build_file}" MATCHES "pool = custom_command_pool")
  set(RunCMake_TEST_FAILED "Log file:\n ${log}\ndoes not have expected line: pool = custom_command_pool")
endif()
if(NOT "${build_file}" MATCHES "pool = custom_target_pool")
  set(RunCMake_TEST_FAILED "Log file:\n ${log}\ndoes not have expected line: pool = custom_target_pool")
endif()
