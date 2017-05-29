#
# Set the name of the project and target:
#
SET(DEAL_II_QC_PACKAGE_NAME "deal.II-qc")
SET(TARGET "qc")
SET(TARGETLIB ${DEAL_II_QC_PACKAGE_NAME})


INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/include)

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
  source/atom/atom_handler.cc
  source/atom/data_out_atom_data.cc
  source/atom/parse_atom_data.cc
  source/atom/sampling/cluster_weights_by_base.cc
  source/atom/sampling/cluster_weights_by_cell.cc
  source/atom/sampling/cluster_weights_by_lumped_vertex.cc
  source/atom/sampling/cluster_weights_by_vertex.cc
  source/configure/configure_qc.cc
  source/configure/geometry/geometry_base.cc
  source/configure/geometry/geometry_box.cc
  source/configure/geometry/geometry_gmsh.cc
  source/potentials/pair_base.cc
  source/potentials/pair_coul_wolf.cc
  source/potentials/pair_lj_cut.cc
  source/core/qc.cc
  ${CMAKE_CURRENT_BINARY_DIR}/version.cc
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

# Run macro to get Git info:
DEAL_II_QUERY_GIT_INFORMATION()
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/source/version.cc.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.cc)

#
# PROJECT has to be called after DEAL_II_INITIALIZE_CACHED_VARIABLES
# otherwise CMake will have already set the compiler and compiler
# flags.
#
PROJECT(${TARGET})

#
# Information about relative paths for installation.
#
SET(DEAL_II_QC_EXECUTABLE_RELDIR     "bin"       )
SET(DEAL_II_QC_INCLUDE_RELDIR        "include"   )
SET(DEAL_II_QC_LIBRARY_RELDIR        "lib"       )
SET(DEAL_II_QC_PROJECT_CONFIG_RELDIR "${DEAL_II_QC_LIBRARY_RELDIR}/cmake/${DEAL_II_QC_PACKAGE_NAME}")

#
# Load in version info and export it
#
FILE(STRINGS "${CMAKE_SOURCE_DIR}/VERSION" DEAL_II_QC_PACKAGE_VERSION LIMIT_COUNT 1)

INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
  "${CMAKE_BINARY_DIR}/${DEAL_II_QC_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${DEAL_II_QC_PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Configure a cmake fragment that plugins can use to
# set up compiler flags, include paths, etc to compile plugin
CONFIGURE_FILE(
  ${CMAKE_SOURCE_DIR}/cmake/${DEAL_II_QC_PACKAGE_NAME}Config.cmake.in
  ${CMAKE_BINARY_DIR}/${DEAL_II_QC_PACKAGE_NAME}Config.cmake
  @ONLY
)

#
# Specify a target:
#
ADD_LIBRARY(${TARGETLIB} SHARED
    ${TARGET_SRC}
  )

ADD_EXECUTABLE(${TARGET}
    source/main.cc
  )

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

#
# Doxygen
#
find_package(Doxygen)
IF(NOT DOXYGEN_FOUND)
    MESSAGE("-- Doxygen was not found. Documentation will not be available.")
ENDIF()
if(DOXYGEN_FOUND)
  SET( doxyfile_in          ${CMAKE_CURRENT_SOURCE_DIR}/doc/options.dox.in    )
  SET( doxyfile             ${CMAKE_CURRENT_BINARY_DIR}/options.dox           )
  SET( doxy_input           ${CMAKE_CURRENT_SOURCE_DIR}/include               ) # Pasted into ${doxyfile_in} in INPUT
  SET( doxy_input_root      ${CMAKE_CURRENT_SOURCE_DIR}/doc                   ) # Posted into ${doxyfile_in}
  SET( doxy_extra_files     "${CMAKE_CURRENT_SOURCE_DIR}/doc/mainpage.dox"    ) # Pasted into ${doxyfile_in} in INPUT; separate with spaces
  SET( doxy_output_root     ${CMAKE_CURRENT_BINARY_DIR}/doc/realgemdft        ) # Pasted into ${doxyfile_in} as HTML_OUTPUT
  SET( doxy_project_name    "")

  #-- Configure the Template Doxyfile for our specific project
  configure_file(${doxyfile_in} ${doxyfile} @ONLY)
  #-- Add a custom target to run Doxygen when ever the project is built
  #-- First command creates a directory in current build folder.
  #   Second command runs Doxygen.
  add_custom_target (doc ${CMAKE_COMMAND} -E make_directory doc
                     COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     COMMENT "Generating API documentation with Doxygen" VERBATIM
                     )
endif(DOXYGEN_FOUND)


# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
# default 600sec is not enough for some tests, increase:
SET (TEST_TIME_LIMIT 900)
#
# tests for core functionality
ADD_SUBDIRECTORY(tests/atom)
ADD_SUBDIRECTORY(tests/atom/sampling)
ADD_SUBDIRECTORY(tests/configure)
ADD_SUBDIRECTORY(tests/core)
ADD_SUBDIRECTORY(tests/potentials)
ADD_SUBDIRECTORY(tests/utilities)


#
# installation target.
# use with -DCMAKE_INSTALL_PREFIX=/path/to/
#
# 1. Install defined library target
install (TARGETS ${TARGETLIB}
         LIBRARY DESTINATION ${DEAL_II_QC_LIBRARY_RELDIR})
# 2. Install the binary
install (TARGETS ${TARGET}
         RUNTIME DESTINATION ${DEAL_II_QC_EXECUTABLE_RELDIR})
# 3. Install content of include dir
install (DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
         DESTINATION ${DEAL_II_QC_INCLUDE_RELDIR})
# 4. Install cmake configuration files in lib/cmake/deal.II-qc folder
install (FILES
         "${CMAKE_CURRENT_BINARY_DIR}/${DEAL_II_QC_PACKAGE_NAME}Config.cmake"
         "${CMAKE_CURRENT_BINARY_DIR}/${DEAL_II_QC_PACKAGE_NAME}ConfigVersion.cmake"
         DESTINATION ${DEAL_II_QC_PROJECT_CONFIG_RELDIR})
