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

    // TODO: Remove local_index from Atom class
    /**
     * Local atom index of this atom within the cell it is associated with.
     *
     * @note During constructing FEValues object for the cell associated to this
     * atom, the local_index of this atom can be used as the index for the
     * quadrature point corresponding to this atom. It is convenient to have
     * the index of the quadrature point while calling the value function of
     * the FEValues object.
     */
    unsigned int local_index;

    /**
     * Atom species type.
     */
    types::atom_type type;

    // TODO: Remove cluster_weight from Atom class
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

    // TODO: Remove reference_position from Atom class.
    /**
     * Position of the atom in reference coordinates of a cell it belongs to.
     */
    Point<spacedim> reference_position;

    // TODO: Remove parent_cell from Atom class
    /**
     * Iterator to a cell which owns this atom.
     */
    typename DoFHandler<spacedim>::active_cell_iterator parent_cell;

  };


} // namespace dealiiqc

#endif // __dealii_qc_atom_h
