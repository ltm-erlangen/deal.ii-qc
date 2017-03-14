
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

namespace dealiiqc
{
  namespace types
  {
    // typedefs

    // When the number of atoms is large and exceeds maximum value
    // allowed with an `unsgined int` (which is system dependent limit)
    // Allow building with 64bit index space.
    typedef  dealii::types::global_dof_index global_atom_index;

    // TODO: Use of correct charge units; Use charge_t for book keeping.
    // For now just use float (float takes less time to compute)
    // (charge of atoms doesn't need high precision)
    typedef float charge;

  } //typedefs

  // Placeholder for some utility functions

}

#endif /* __dealii_qc_utility_h */
