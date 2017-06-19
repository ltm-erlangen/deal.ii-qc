#ifndef __dealii_qc_atom_h
#define __dealii_qc_atom_h

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II-qc/utilities.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class for atoms embedded in a <tt>spacedim</tt>-dimensional space.
   */
  template <int spacedim>
  struct Atom
  {

    /**
     * Global atom index of this atom.
     */
    types::global_atom_index global_index;

    /**
     * Atom species type.
     */
    types::atom_type type;

    /**
     * Current position in real space.
     */
    Point<spacedim> position;

  };


} // namespace dealiiqc

#endif // __dealii_qc_atom_h
