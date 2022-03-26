
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

MESSAGE(STATUS "Setting up benchmarks")

PROJECT(Benchmarks)

FIND_PACKAGE(deal.II-qc QUIET HINTS ${deal.II-qc_DIR} ${DEAL_II_QC_DIR} $ENV{DEAL_II_QC_DIR})

IF (NOT deal.II-qc_FOUND)
  MESSAGE(FATAL_ERROR "\n"
        "Could not find a valid deal.II-qc build/installation directory. "
        "Please specify the directory where you are building deal.II-qc by passing\n"
        "   -D DEAL_II_QC_DIR=<path to DEAL_II_QC>\n"
        "to cmake or by setting the environment variable DEAL_II_QC_DIR in your shell "
        "before calling cmake. See the section 'How to write a plugin' in the "
        "manual for more information.")
ENDIF ()
MESSAGE( STATUS "deal.II-qc_FOUND at: "        ${DEAL_II_QC_PATH} )

FIND_PACKAGE(deal.II ${DEAL_II_VERSION_MAJOR}.${DEAL_II_VERSION_MINOR}.${DEAL_II_VERSION_SUBMINOR} 
             QUIET REQUIRED HINTS ${DEAL_II_PATH} ${DEAL_II_DIR}
             )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a version of deal.II "
    " that was used to build deal.II-qc. ***\n\n"
    )
ENDIF()
MESSAGE( STATUS "deal.II_FOUND at: "        ${DEAL_II_PATH} )

#
# Find Google microbenchmark support library
#
FIND_PACKAGE(benchmark REQUIRED
  HINTS ${benchmark_DIR} ${BENCHMARK_DIR} ../ ../../ $ENV{benchmark_DIR}
)

#
# Find Boost library
#
FIND_PACKAGE(Boost REQUIRED)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

FILE(GLOB BENCHMARKS "*.bench.cc")

FOREACH(BENCHMARK ${BENCHMARKS})
  GET_FILENAME_COMPONENT(BENCH_FILE "${BENCHMARK}" NAME)
  GET_FILENAME_COMPONENT(SUB_TARGET "${BENCHMARK}" NAME_WE)

  ADD_EXECUTABLE(${SUB_TARGET} ${BENCH_FILE})
  DEAL_II_SETUP_TARGET(${SUB_TARGET})
  TARGET_LINK_LIBRARIES(${SUB_TARGET} benchmark::benchmark)
ENDFOREACH()
