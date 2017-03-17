
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

namespace dealiiqc
{
  /**
   * Custom types used inside dealiiqc.
   */
  namespace types
  {
    /**
     * The type used for global indices of atoms.
     * In order to have 64-bit unsigned integers (more than 4 billion),
     *  build deal.II with support for 64-bit integers.
     *  The data type always indicates an unsigned integer type.
     */
    typedef  dealii::types::global_dof_index global_atom_index;

    // TODO: Use of correct charge units; Use charge_t for book keeping.
    /**
     * The type used for storing charge of the atom.
     * Computations with charge of atoms don't need high precision.
     */
    typedef float charge;

    /**
     * The type used for identifying atom types. The enumeration starts
     * from 0.
     */
    typedef unsigned char atom_type;

  } //typedefs

/**
 * Make sure that sscanf doesn't pickup spaces as unsigned char
 * while parsing atom data stream.
 */
#define UC_SCANF_STR "%hhu"

}

#endif /* __dealii_qc_utility_h */
