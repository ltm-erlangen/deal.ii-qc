
#
# Install the following git-hooks if .git directory exists:
# pre-commit
# pre-push
#
# If the above git-hooks are already present in .git directory,
# they will not be overwritten.
#
IF(EXISTS ${PROJECT_SOURCE_DIR}/.git)
  #
  # Install pre-commit git hook to a temporary folder to populate
  # cmake variables if required.
  #
  IF(NOT EXISTS ${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit)

    CONFIGURE_FILE(
      ${CMAKE_SOURCE_DIR}/cmake/git-hooks/pre-commit.in
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pre-commit
      @ONLY
    )
  ENDIF()
  #
  # Install pre-push git hook.
  #
  IF(NOT EXISTS ${PROJECT_SOURCE_DIR}/.git/hooks/pre-push)

    CONFIGURE_FILE(
      ${CMAKE_SOURCE_DIR}/cmake/git-hooks/pre-push.in
      ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pre-push
      @ONLY
    )
  ENDIF()


  #
  # Install pre-commit git hook in .git/hooks folder
  # (with file execution permissions).
  #
  IF(NOT EXISTS ${CMAKE_SOURCE_DIR}/.git/hooks/pre-commit)
    FILE(
      COPY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pre-commit
      DESTINATION ${CMAKE_SOURCE_DIR}/.git/hooks
      FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
      GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
  ENDIF()
	#
	# Install pre-push git hook in .git/hooks folder
	# (with file execution permissions).
	#
  IF(NOT EXISTS ${CMAKE_SOURCE_DIR}/.git/hooks/pre-push)
    FILE(
      COPY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pre-push
      DESTINATION ${CMAKE_SOURCE_DIR}/.git/hooks
      FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
      GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
  ENDIF()

ENDIF()
