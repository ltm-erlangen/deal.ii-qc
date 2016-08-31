# Set the name of the project and target:
SET(TARGET "qc")
SET(TARGETLIB "qc_dealii")

INCLUDE_DIRECTORIES(
 ${CMAKE_SOURCE_DIR}/include
)

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#  FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#  SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  source/qc/core.cc
  )

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

#
# PROJECT has to be called after DEAL_II_INITIALIZE_CACHED_VARIABLES
# otherwise CMake will have already set the compiler and compiler
# flags.
#
PROJECT(${TARGET})

#
# Specify a target:
#
ADD_LIBRARY(${TARGETLIB} SHARED
    ${TARGET_SRC}
  )

ADD_EXECUTABLE(${TARGET}
    source/main.cc
  )

# A note on RPath:
# -DCMAKE_INSTALL_RPATH=xyz ...
#
# results in "xyz" appearing as the RPATH in the final binaries *after*
# the installation, i.e. during "make install", executables and shared
# libraries are relinked, so the CMAKE_INSTALL_RPATH is finally included
# in the installed binaries. If you want to have the CMAKE_INSTALL_RPATH
# already available right after building, e.g. to test your software from
# within CMAKE_BINARY_DIR, you should set CMAKE_BUILD_WITH_INSTALL_RPATH
# to TRUE.
SET (CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# DEAL_II_SETUP_TARGET will set up all necessary include paths,
# preprocessor definitions and the link interface:
#
DEAL_II_SETUP_TARGET(${TARGETLIB})
DEAL_II_SETUP_TARGET(${TARGET})

TARGET_LINK_LIBRARIES(${TARGET}
    ${TARGETLIB}
  )

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/include)

#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)


# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
# default 600sec is not enough for some tests, increase:
SET (TEST_TIME_LIMIT 900)
#
# ADD_SUBDIRECTORY(tests/...)
#
# installation target.
# use with -DCMAKE_INSTALL_PREFIX=/path/to/
#
# 1. Install defined library target
install (TARGETS ${TARGETLIB}
         LIBRARY DESTINATION lib)
# 2. Install the binary
install (TARGETS ${TARGET}
         RUNTIME DESTINATION bin)
# 3. Install content of include dir
install (DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
         DESTINATION include)
