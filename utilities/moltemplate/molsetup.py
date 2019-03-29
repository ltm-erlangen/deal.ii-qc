#!/usr/bin/env python

""" This python scripts generates moltemplate files and autogenerates
    atom data files for atomistic simulations (lammps ready).

    Vishal Boddu May 2017
"""
import os
import shutil
import subprocess
from argparse import ArgumentParser


#-------------------------------------------------PARSE COMMAND LINE ARGUMENTS
PARSER = ArgumentParser(description='Follow the documentation of '
                                    'the command line arguments below.')
PARSER.add_argument('lt_file_name', metavar='lt_file_name',
                    help='is the name of the atom data file to be generated')
PARSER.add_argument('in_file_lt', metavar='in_file_lt',
                    help='is the path to input lt file '
                         '(including complete file name) '
                         'that contains the description of the '
                         'unit cell `in_cell_lt` in moltemplate format.')
PARSER.add_argument('in_cell_lt', metavar='in_cell_lt',
                    help='is the unit cell being replicated ')
PARSER.add_argument('s', metavar='s', type=float,
                    help='amount by which the a unit cell is to be scaled')
PARSER.add_argument('a', metavar='a', type=float,
                    help='is the crystal lattice constant in X direction')
PARSER.add_argument('b', metavar='b', type=float,
                    help='is the crystal lattice constant in Y direction')
PARSER.add_argument('c', metavar='c', type=float,
                    help='is the crystal lattice constant in Z direction')
PARSER.add_argument('nx', metavar='nx', type=int,
                    help='is the number of replicas in X direction in the RVE')
PARSER.add_argument('ny', metavar='ny', type=int,
                    help='is the number of replicas in Y direction in the RVE')
PARSER.add_argument('nz', metavar='nz', type=int,
                    help='is the number of replicas in Z direction in the RVE')

ARGS = PARSER.parse_args()

IN_FILE_LT = ARGS.in_file_lt
IN_CELL_LT = ARGS.in_cell_lt
LT_FILE_NAME = ARGS.lt_file_name + ".lt"

S = ARGS.s
A = ARGS.a
B = ARGS.b
C = ARGS.c
NX = ARGS.nx
NY = ARGS.ny
NZ = ARGS.nz

LX = str(float(A) * int(NX))
LY = str(float(B) * int(NY))
LZ = str(float(C) * int(NZ))

#-------------------------------------------------TO COPY INTO MOLTEMPLATE FILE
RAW_TXT = """
# Generates atom.data describing all the atoms in a SYSTEM that
# consist of unit cells described by in_cell_lt.
# This file aids in creation of atom data files, through moltemplate
# compatible with read_data in LAMMPS. \n
import "%s" \n \n
%s.scale( %s ) \n
# Periodic boundary conditions:
write_once("Data Boundary") {
   0.0  %s  xlo xhi
   0.0  %s  ylo yhi
   0.0  %s  zlo zhi
}
SYSTEM = new %s[%s].move(  %s, 0.0, 0.0)
               [%s].move( 0.0,  %s, 0.0)
               [%s].move( 0.0,  0.0, %s)
"""%(IN_FILE_LT, IN_CELL_LT, S, LX, LY, LZ, IN_CELL_LT, NX, A, NY, B, NZ, C)

#-------------------------------------------------CREATE MOLTEMPLATE FILE
LT_FILE = open(LT_FILE_NAME, "w")
LT_FILE.write(RAW_TXT)
LT_FILE.close()

#-------------------------------------------------GENERATE DATA FILE FOR LAMMPS
subprocess.check_call(["moltemplate.sh", LT_FILE_NAME, "-nocheck"])

#-------------------------------------------------DELETE TEMP FILES
IN_FILE_NAME = LT_FILE_NAME.replace("lt", "in")
os.remove(LT_FILE_NAME)
os.remove(IN_FILE_NAME)
shutil.rmtree("output_ttree")
