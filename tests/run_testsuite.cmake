
########################################################################
#                                                                      #
#                             Test setup:                              #
#                                                                      #
########################################################################

#
# This is the ctest script for running and submitting build and regression
# tests.
#
# Invoke it in a _build directory_ (or designated build directory) via:
#
#   ctest -S <...>/run_testsuite.cmake
#
# The following configuration variables can be overwritten with
#
#   ctest -D<variable>=<value> [...]
#
#
#   CTEST_SOURCE_DIRECTORY
#     - The source directory
#       Note: This is _not_ the test directory ending in "[...]/tests"
#
#   CTEST_BINARY_DIRECTORY
#     - The designated build directory (already configured, empty, or non
#       existent - see the information about TRACKs what will happen)
#     - If unspecified the current directory is used. If the current
#       directory is equal to CTEST_SOURCE_DIRECTORY or the "tests"
#       directory, an error is thrown.
#
#   CTEST_CMAKE_GENERATOR
#     - The CMake Generator to use (e.g. "Unix Makefiles", or "Ninja", see
#       $ man cmake)
#     - If unspecified the generator of a configured build directory will
#       be used, otherwise "Unix Makefiles".
#
#   TRACK
#     - The track the test should be submitted to. Defaults to
#       "Experimental". Possible values are:
#
#       "Experimental"     - all tests that are not specifically "build" or
#                            "regression" tests should go into this track
#
#       "Build Tests"      - Build tests that configure and build in a
#                            clean directory and run the build tests
#                            "build_tests/*"
#
#       "Nightly"          - Reserved for nightly regression tests for
#                            build bots on various architectures
#
#       "Regression Tests" - Reserved for the regression tester
#
#   DESCRIPTION
#     - A string that is appended to CTEST_BUILD_NAME
#
#   GIT_UPDATE_ARGS
#     - Arguments for a custom Git update command (optional). For example
#       one can use "show" to skip update altogether but still write Git info
#       in Update.xml file
#
# Furthermore, the following variables controlling the testsuite can be set
# and will be automatically handed down to cmake:
#
#   TEST_DIFF
#   TEST_TIME_LIMIT
#   TEST_PICKUP_REGEX
#   TEST_OVERRIDE_LOCATION
#
# For details, consult the ./README file.
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
MESSAGE("-- This is CTest ${CMAKE_VERSION}")

#
# TRACK: Default to Experimental:
#

IF("${TRACK}" STREQUAL "")
  SET(TRACK "Experimental")
ENDIF()

IF( NOT "${TRACK}" STREQUAL "Experimental"
    AND NOT "${TRACK}" STREQUAL "Build Tests"
    AND NOT "${TRACK}" STREQUAL "Nightly"
    AND NOT "${TRACK}" STREQUAL "Regression Tests" )
  MESSAGE(FATAL_ERROR "
Unknown TRACK \"${TRACK}\" - see the manual for valid values.
"
    )
ENDIF()

MESSAGE("-- TRACK:                  ${TRACK}")

#
# CTEST_SOURCE_DIRECTORY:
#

IF("${CTEST_SOURCE_DIRECTORY}" STREQUAL "")
  #
  # If CTEST_SOURCE_DIRECTORY is not set we just assume that this script
  # resides in the top level source directory
  #
  SET(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  # The code above set the source directory of that of the run_testsuite.cmake
  # script, but we need one level higher
  IF ("${CTEST_SOURCE_DIRECTORY}" MATCHES "/tests")
    SET(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)
  ENDIF()

  IF(NOT EXISTS ${CTEST_SOURCE_DIRECTORY}/CMakeLists.txt)
    MESSAGE(FATAL_ERROR "
Could not find a suitable source directory. Please, set
CTEST_SOURCE_DIRECTORY manually to the appropriate source directory.
"
      )
  ENDIF()
ENDIF()

MESSAGE("-- CTEST_SOURCE_DIRECTORY: ${CTEST_SOURCE_DIRECTORY}")

#
# CTEST_BINARY_DIRECTORY:
#

IF("${CTEST_BINARY_DIRECTORY}" STREQUAL "")
  #
  # If CTEST_BINARY_DIRECTORY is not set we just use the current directory
  # except if it is equal to CTEST_SOURCE_DIRECTORY in which case we fail.
  #
  SET(CTEST_BINARY_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
ENDIF()

# Make sure that for a build test the directory is empty:
FILE(GLOB _test ${CTEST_BINARY_DIRECTORY}/*)
IF( "${TRACK}" STREQUAL "Build Tests"
    AND NOT "${_test}" STREQUAL "" )
      MESSAGE(FATAL_ERROR "
TRACK was set to \"Build Tests\" which require an empty build directory.
But files were found in \"${CTEST_BINARY_DIRECTORY}\"
"
        )
ENDIF()

MESSAGE("-- CTEST_BINARY_DIRECTORY: ${CTEST_BINARY_DIRECTORY}")

#
# CTEST_CMAKE_GENERATOR:
#

# Query Generator from build directory (if possible):
IF(EXISTS ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt)
  FILE(STRINGS ${CTEST_BINARY_DIRECTORY}/CMakeCache.txt _generator
    REGEX "^CMAKE_GENERATOR:"
    )
  STRING(REGEX REPLACE "^.*=" "" _generator ${_generator})
ENDIF()

IF("${CTEST_CMAKE_GENERATOR}" STREQUAL "")
  IF(NOT "${_generator}" STREQUAL "")
    SET(CTEST_CMAKE_GENERATOR ${_generator})
  ELSE()
    # default to "Unix Makefiles"
    SET(CTEST_CMAKE_GENERATOR "Unix Makefiles")
  ENDIF()
ELSE()
  # ensure that CTEST_CMAKE_GENERATOR (that was apparantly set) is
  # compatible with the build directory:
  IF( NOT "${CTEST_CMAKE_GENERATOR}" STREQUAL "${_generator}"
      AND NOT "${_generator}" STREQUAL "" )
    MESSAGE(FATAL_ERROR "
The build directory is already set up with Generator \"${_generator}\", but
CTEST_CMAKE_GENERATOR was set to a different Generator \"${CTEST_CMAKE_GENERATOR}\".
"
     )
  ENDIF()
ENDIF()

MESSAGE("-- CTEST_CMAKE_GENERATOR:  ${CTEST_CMAKE_GENERATOR}")

#
# CTEST_SITE:
#

FIND_PROGRAM(HOSTNAME_COMMAND NAMES hostname)
IF(NOT "${HOSTNAME_COMMAND}" MATCHES "-NOTFOUND")
  EXEC_PROGRAM(${HOSTNAME_COMMAND} OUTPUT_VARIABLE _hostname)
  SET(CTEST_SITE "${_hostname}")
ELSE()
  # Well, no hostname available. What about:
  SET(CTEST_SITE "BobMorane")
ENDIF()

MESSAGE("-- CTEST_SITE:             ${CTEST_SITE}")

#
# Assemble configuration options, we need it now:
#

IF("${TRACK}" STREQUAL "Build Tests")
  SET(TEST_PICKUP_REGEX "^build_tests")
ENDIF()

# Pass all relevant variables down to configure:
GET_CMAKE_PROPERTY(_variables VARIABLES)

#
# CTEST_BUILD_NAME:
#

IF(NOT EXISTS ${CTEST_BINARY_DIRECTORY}/detailed.log)
  MESSAGE(FATAL_ERROR "could not find detailed.log")
ENDIF()

# Append compiler information to CTEST_BUILD_NAME:
# and query Git info:
IF(EXISTS ${CTEST_BINARY_DIRECTORY}/detailed.log)
  FILE(STRINGS ${CTEST_BINARY_DIRECTORY}/detailed.log _compiler_id
    REGEX "CMAKE_CXX_COMPILER:"
    )
  STRING(REGEX REPLACE
    "^.*CMAKE_CXX_COMPILER:        \(.*\) on platform.*$" "\\1"
    _compiler_id ${_compiler_id}
    )
  STRING(REGEX REPLACE "^\(.*\) .*$" "\\1" _compiler_name ${_compiler_id})
  STRING(REGEX REPLACE "^.* " "" _compiler_version ${_compiler_id})
  STRING(REGEX REPLACE " " "-" _compiler_id ${_compiler_id})
  IF( NOT "${_compiler_id}" STREQUAL "" OR
      _compiler_id MATCHES "CMAKE_CXX_COMPILER" )
    SET(CTEST_BUILD_NAME "${_compiler_id}")
  ENDIF()

  FILE(STRINGS ${CTEST_BINARY_DIRECTORY}/detailed.log _build_line
       REGEX "CMAKE_BUILD_TYPE:"
      )
  STRING(REGEX REPLACE "^.*#.*CMAKE_BUILD_TYPE: *(.*)$" "\\1" _build_type ${_build_line})

  FILE(STRINGS ${CTEST_BINARY_DIRECTORY}/detailed.log _rev_line
       REGEX "GIT_SHORTREV:"
      )
  STRING(REGEX REPLACE "^.*#.*GIT_SHORTREV: *(.*)$" "\\1" _git_WC_REV ${_rev_line})

  FILE(STRINGS ${CTEST_BINARY_DIRECTORY}/detailed.log _branch_line
       REGEX "GIT_BRANCH:"
      )
  STRING(REGEX REPLACE "^.*#.*GIT_BRANCH: *(.*)$" "\\1" _git_WC_BRANCH ${_branch_line})

ENDIF()

SET(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-${_git_WC_BRANCH}-${_build_type}")

#
# Write revision log:
#

FILE(WRITE ${CTEST_BINARY_DIRECTORY}/revision.log
"###
#
#  Git information:
#    Branch: ${_git_WC_BRANCH}
#    Commit: ${_git_WC_REV}
#
###"
  )

#
# Append DESCRIPTION string to CTEST_BUILD_NAME:
#

IF(NOT "${DESCRIPTION}" STREQUAL "")
  SET(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-${DESCRIPTION}")
ENDIF()

MESSAGE("-- CTEST_BUILD_NAME:       ${CTEST_BUILD_NAME}")

#
# Declare files that should be submitted as notes:
#

SET(CTEST_NOTES_FILES
  ${CTEST_BINARY_DIRECTORY}/revision.log
  ${CTEST_BINARY_DIRECTORY}/detailed.log
  )

MESSAGE("-- CMake Options:          ${_options}")

########################################################################
#                                                                      #
#                          Run the testsuite:                          #
#                                                                      #
########################################################################

CTEST_START(Experimental TRACK ${TRACK})

FIND_PROGRAM(CTEST_GIT_COMMAND NAMES git)
MESSAGE("-- Running CTEST_UPDATE()")
IF (NOT "${GIT_UPDATE_ARGS}" STREQUAL "")
  SET(CTEST_GIT_UPDATE_CUSTOM "${CTEST_GIT_COMMAND}" "${GIT_UPDATE_ARGS}")
ENDIF()
CTEST_UPDATE()

MESSAGE("-- Running CTEST_CONFIGURE()")
CTEST_CONFIGURE(OPTIONS "${_options}" RETURN_VALUE _res)

IF("${_res}" STREQUAL "0")
  # Only run the build stage if configure was successful:

  MESSAGE("-- Running CTEST_BUILD()")
  CTEST_BUILD(TARGET ${MAKEOPTS} NUMBER_ERRORS _res)

  IF("${_res}" STREQUAL "0")
    # Only run tests if the build was successful:

    MESSAGE("-- Running CTEST_TEST()")
    CTEST_TEST()
  ENDIF()
ENDIF()

# Using time at which the test is submitted as StartDateTime and EndDateTime
# to silence error. 

MESSAGE("-- Running CTEST_SUBMIT()")
CTEST_SUBMIT(RETURN_VALUE _res)

IF("${_res}" STREQUAL "0")
  MESSAGE("-- Submission successful. Goodbye!")
ENDIF()

