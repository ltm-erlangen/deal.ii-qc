
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

namespace dealiiqc
{
  namespace typedefs
  {
    // typedefs

    // When the number of atoms is large and exceeds maximum value
    // allowed with an `unsgined int` (which is system dependent limit)
    // Allow building with 64bit index space.
    typedef  dealii::types::global_dof_index global_atom_index;

    // TODO: Use of correct charge units; Use charge_t for book keeping.
    // For now just use float (float takes less time to compute)
    // (charge of atoms doesn't need high precision)
    typedef float charge_t;

  } //typedefs

  // Some utility functions

  /**
   *  Check if a container's first few elements are exactly
   *  the same as another container
   */
  template<class Container>
  bool begins_with(const Container &input, const Container &match)
  {
    return input.size() >= match.size()
           && std::equal(match.begin(), match.end(), input.begin());
  }


}

#endif /* __dealii_qc_utility_h */
