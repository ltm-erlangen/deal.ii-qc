
from __future__ import print_function

import argparse
import time
import distutils.spawn
import distutils.file_util

import fnmatch
import multiprocessing
import os
import Queue
import re
import shutil
import subprocess
import threading

from tempfile import mkdtemp, mkstemp
from filecmp  import cmp


def parse_arguments ():
  """
  Argument parser.
  """
  parser = argparse.ArgumentParser("Run clang-format on a list of files "
                                   "in a given list of folders "
                                   "having a given file extension."
                                   "This program requires "
                                   "clang-format version 6.0.")

  parser.add_argument("-b", "--clang-format-binary", metavar="PATH",
                      default=distutils.spawn.find_executable("clang-format"))

  parser.add_argument("--regex", default="*.cc,*.h",
                      help="Reggular expression (regex) to filter files on "
                      "which clang-format is applied.")

  parser.add_argument("--dry-run", default=False,
                      help="If dry-run is set to True, a list of file names "
                      "are written out, without actually formatting them.")

  parser.add_argument("-dirs", "--directories", default="include,source,tests",
                      help="Comma-delimited list of directories to work on.",)

  parser.add_argument("-j", metavar="THREAD_COUNT", type=int, default=0,
                      help="Number of clang-format instances to be run in parallel.")

  args = parser.parse_args()

  if not args.clang_format_binary:
    os.environ["PATH"] += ':' + os.getcwd() + "/utilities/programs/clang-6/bin"
    args.clang_format_binary = distutils.spawn.find_executable("clang-format")

  return args

def sanity_checks(args):
  """
  Make sure that the following conditions are satisfied.
  Throw suitable recommendations if any of the conditions are not satisfied:
  1. The current working directory is the top-level directory of deal.II-qc.
  2. A suitable version of ClangFormat should have been installed.
  """
  #
  # Assert that the current directory is the top-level directory by
  # by looking up if include, source, tests and utilities folders exist
  # in the current working directory.
  #
  assert (os.path.exists("include") and os.path.exists("source") and
          os.path.exists("tests")   and os.path.exists("utilities"))

  #
  # Check if clang-format is already installed.
  #
  try:
    clang_format_version = subprocess.check_output([args.clang_format_binary,
                                                   '--version'])
    version_number = re.search(r'Version\s*([\d.]+)',
                               clang_format_version,
                               re.IGNORECASE).group(1)
    print ("Found clang-format Version: ", version_number)
    assert (version_number == "6.0.0" or version_number == "6.0.1")
  except OSError:
    print("***"
          "***   No clang-format program found."
          "***"
          "***   You can run"
          "***       'python utilities/download_clang_format'"
          "***   or"
          "***       'python utilities/compile_clang_format'"
          "***   to install a compatible binary into './utilities/programs'."
          "***")
    raise
  except subprocess.CalledProcessError as e:
    print(e.output)
    raise

def format_file(args, task_queue, temp_dir):
  """
  A given thread worker takes out sourcecode files, one at a time,
  out of the task_queue and tries to apply clang-format to format code on them.
  Each thread continuously attempts to empty the task_queue.
  If args.dry_run is set to True, only report the file names of files that
  are found with incorrect formatting without overriding them.

  Arguments:
  args           -- arguments provided to this program
  task_queue     -- a queue of sourcecode files (or file names) to be formatted
  full_file_name -- the file name of the file to be formatted/indented
  temp_dir       -- temporary directory to dump temporary files used to diff
  """
  while True:
    #
    # Get a file name from the list of sourcecode files in task_queue
    #
    full_file_name = task_queue.get()
    #
    # Get file name ignoring full path of the directory its in.
    #
    file_name = os.path.basename(full_file_name)
    #
    # Generate a temporary file and copy the contents of the given file.
    #
    _, temp_file_name = mkstemp (dir=temp_dir,
                                 prefix=file_name+'.',
                                 suffix='.tmp')

    shutil.copyfile (full_file_name, temp_file_name)
    #
    # Prepare command line statement to be executed by a subprocess call
    # to apply formatting to file_name.
    #
    apply_clang_format_str = args.clang_format_binary + ' '   + \
                             full_file_name           + " > " + temp_file_name
    try:
      subprocess.call (apply_clang_format_str, shell=True)
    except OSError:
      print("Unknown OSError")
      raise
    except subprocess.CalledProcessError as e:
      print(e.output)
      raise
    #
    # Compare the original file and the formatted file from clang-format.
    # If it's a dry run and if files differ, write out that the original file
    # is not formatted correctly.
    # Otherwise override the original file with formatted content.
    #
    if not cmp(full_file_name, temp_file_name):
      if args.dry_run:
        os.remove (temp_file_name)
      else:
        shutil.move (temp_file_name, full_file_name)
    #
    # Indicate that the current file name is processed by the current thread.
    # Once task_done() is called by a thread, the total count of
    # unfinished tasks goes down by one.
    #
    task_queue.task_done()

def process (args):
  """
  Collect all files found in the directories list matching with
  a given regex (regular expression) with n_threads number of threads
  in parallel.
  """
  tmpdir = mkdtemp()

  n_threads = args.j
  if n_threads == 0:
    n_threads = multiprocessing.cpu_count()-1
  print ("Number of threads picked up:", n_threads)

  #
  # Create n_threads number of queues, one for each thread.
  #
  task_queue = Queue.Queue(n_threads)

  #
  # Start n_threads number of thread workers that will execute
  # a target with given arguments.
  #
  for _ in range (n_threads):
    thread_worker = threading.Thread(target=format_file,
                                      args=(args, task_queue, tmpdir))
    thread_worker.daemon = True
    thread_worker.start()

  #
  # Gather all the files that are needed to be formatted in task_queue.
  # Look through all directories, recursively, and find all files
  # that match the given regex.
  #
  for directory in args.directories.split(','):
    for dirpath, _, filenames in os.walk(directory):
      for pattern in args.regex.split(','):
        for file_name in fnmatch.filter(filenames, pattern):
          task_queue.put(os.path.join(dirpath, file_name))
  #
  # Blocks (some) threads until all the threads finished their tasks.
  # Works similar to MPI_Barrier().
  # In other words, threads wait untill all the tasks in task_queue
  # have finshed.
  #
  task_queue.join()

  shutil.rmtree(tmpdir)


start = time.time()

if __name__ == "__main__":
  args = parse_arguments()
  sanity_checks(args)
  process(args)

finish = time.time()

print("Finished code formatting in: ", (finish-start), " seconds.")
