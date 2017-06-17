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

    // TODO: Remove cluster_weight from Atom class. Keeping this so
    //       cluster_weights_by_base like classes compile.
    /**
     * Contribution to energy calculations in terms of cluster weight.
     * All the cluster_atoms have non-zero @p cluster_weight.
     * Any atom that is located inside a cluster is a cluster_atom i.e.,
     * an atom is in a cluster if it's within a distance of @see cluster_radius
     * to any vertex.
     */
    double cluster_weight;

    /**
     * Current position in real space.
     */
    Point<spacedim> position;

    // TODO: Remove reference_position from Atom class. Keeping this so
    //       cluster_weights_by_base like classes compile.
    /**
     * Position of the atom in reference coordinates of a cell it belongs to.
     */
    Point<spacedim> reference_position;

  };


} // namespace dealiiqc

#endif // __dealii_qc_atom_h
