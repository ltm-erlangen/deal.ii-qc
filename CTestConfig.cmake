## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME "deal.ii-qc")
set(CTEST_NIGHTLY_START_TIME "00:00:00 CET")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "cdash.ltm.uni-erlangen.de")
set(CTEST_DROP_LOCATION "/submit.php?project=deal.ii-qc")
set(CTEST_DROP_SITE_CDASH TRUE)

SET(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS   100)
SET(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 300)

# number of lines to submit before an error:
SET(CTEST_CUSTOM_ERROR_PRE_CONTEXT            5)
# number of lines to submit after an error:
SET(CTEST_CUSTOM_ERROR_POST_CONTEXT          20)

#
# Coverage options:
#

SET(CTEST_EXTRA_COVERAGE_GLOB
  # These files should have executable lines and therefore coverage:
  # source/**/*.cc
  )

SET(CTEST_CUSTOM_COVERAGE_EXCLUDE
  "/tests/"
  )
