#!/usr/bin/python

""" This python scripts generates moltemplate files and autogenerates
    atom data files for atomistic simulations (lammps ready).

    Vishal Boddu May 2017
"""
import os
import shutil
import subprocess
import sys
import argparse
import re


#-------------------------------------------------PARSE COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description='Follow the documentation of the ' 
                                             'command line arguments below.')
parser.add_argument('in_file_lt', metavar='in_file_lt',
                    help='is the path to input lt file '
                         '(including complete file name) '
                         'that contains the description of the '
                         'unit cell `in_cell_lt` in moltemplate format.')
parser.add_argument('in_cell_lt', metavar='in_cell_lt', 
                    help='is the unit cell being replicated ')
parser.add_argument('s', metavar='s', type=float,
                    help='amount by which the a unit cell is to be scaled')
parser.add_argument('a', metavar='a', type=float,
                    help='is the crystal lattice constant in X direction')
parser.add_argument('b', metavar='b', type=float,
                    help='is the crystal lattice constant in Y direction')
parser.add_argument('c', metavar='c', type=float,
                    help='is the crystal lattice constant in Z direction')
parser.add_argument('nx', metavar='nx', type=int,
                    help='is the number of replicas in X direction in the RVE')
parser.add_argument('ny', metavar='ny', type=int,
                    help='is the number of replicas in Y direction in the RVE')
parser.add_argument('nz', metavar='nz', type=int,
                    help='is the number of replicas in Z direction in the RVE')

args = parser.parse_args()

in_file_lt = args.in_file_lt
in_cell_lt = args.in_cell_lt
s  = args.s
a  = args.a
b  = args.b
c  = args.c
nx = args.nx
ny = args.ny
nz = args.nz

lx = str (float(a) * int(nx))
ly = str (float(b) * int(ny))
lz = str (float(c) * int(nz))

#-------------------------------------------------TO COPY INTO MOLTEMPLATE FILE
raw_txt  = """
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
"""%( in_file_lt, in_cell_lt, s, lx, ly, lz, in_cell_lt, nx, a, ny, b, nz, c )

#-------------------------------------------------CREATE MOLTEMPLATE FILE
lt_file_name = "atom.lt"            #### Choosing a generic name #####
lt_file = open( lt_file_name, "w" )
lt_file.write("%s" % raw_txt )
lt_file.close()

#-------------------------------------------------GENERATE DATA FILE FOR LAMMPS
subprocess.check_call( [ "moltemplate.sh", lt_file_name, "-nocheck" ])

#-------------------------------------------------DELETE TEMP FILES
in_file_name = lt_file_name.replace( "lt", "in" )
os.remove( lt_file_name )
os.remove( in_file_name )
shutil.rmtree( "output_ttree" )
