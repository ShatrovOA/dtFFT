# if(DTFFT_BUILD_JAVA_API)
#   add_subdirectory(src/interfaces/api/java)
#   # add_compile_definitions(DTFFT_BUILD_JAVA_API)
#   # find_package(Java REQUIRED)
#   # include(UseJava)
#   # find_package(JNI REQUIRED)
#   # include_directories(${JNI_INCLUDE_DIRS})
#   # file(GLOB JAVA_SOURCES src/dtfft/*.java)
#   # find_jar(mpiJar mpi PATHS "/usr/local/lib/")
#   # add_jar(dtfft_java
#   #   SOURCES ${JAVA_SOURCES}
#   #   INCLUDE_JARS ${mpiJar}
#   #   OUTPUT_NAME dtfft
#   #   GENERATE_NATIVE_HEADERS java_native_headers)
#   # get_target_property(NATIVE_HEADERS_DIRECTORY java_native_headers NATIVE_HEADERS_DIRECTORY)
#   # target_include_directories(dtfft PRIVATE ${NATIVE_HEADERS_DIRECTORY})
# endif()

  # if (DTFFT_BUILD_JAVA_API)
  #   add_subdirectory(${API_INTERFACES_DIR}/java)
    # if(NOT MPI_JAR_DEST)
    #   message(FATAL_ERROR "MPI_JAR_DEST parameter is not defined")
    # endif()

    # find_package(Java REQUIRED)
    # include(UseJava)
    # set(JAVA_AWT_LIBRARY NotNeeded)
    # set(JAVA_JVM_LIBRARY NotNeeded)
    # find_package(JNI REQUIRED)
    # target_include_directories(dtfft PRIVATE ${JNI_INCLUDE_DIRS})
    # file(GLOB JAVA_SOURCES ${API_INTERFACES_DIR}/java/src/dtfft/*.java)
    # find_jar(mpiJar mpi PATHS ${MPI_JAR_DEST})
    # if(NOT mpiJar)
    #   message(FATAL_ERROR "mpi.jar is not found in ${MPI_JAR_DEST}")
    # endif()

    # add_jar(dtfft_java
    #         SOURCES ${JAVA_SOURCES}
    #         INCLUDE_JARS ${mpiJar}
    #         OUTPUT_NAME dtfft
    #         GENERATE_NATIVE_HEADERS java_native_headers)

    # get_target_property(NATIVE_HEADERS_DIRECTORY java_native_headers NATIVE_HEADERS_DIRECTORY)
    # target_include_directories(dtfft PRIVATE ${NATIVE_HEADERS_DIRECTORY})
    # add_dependencies (dtfft dtfft_java)
    # file(GLOB JAVA_API_SOURCES ${API_INTERFACES_DIR}/java/*.c)
    # set(API_SOURCES "${API_SOURCES};${JAVA_API_SOURCES}")
  # endif()