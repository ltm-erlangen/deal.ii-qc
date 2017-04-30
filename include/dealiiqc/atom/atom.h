#ifndef __dealii_qc_atom_h
#define __dealii_qc_atom_h

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <dealiiqc/utilities.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class for atoms associated with FEM.
   */
  template <int dim>
  struct Atom
  {

    /**
     * Global atom index of this atom.
     */
    types::global_atom_index global_index;

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

    /**
     * Contribution to energy calculations in terms of cluster weight.
     * All the cluster_atoms have non-zero @p cluster_weight.
     * Any atom that is located inside a cluster is a cluster_atom i.e.,
     * an atom is in a cluster if it's within a distance of @see cluster_radius
     * to any vertex.
     */
    double cluster_weight;

    /**
     * Position in real space
     */
    Point<dim> position;

    /**
     * Position of the atom in reference coordinates of a cell it belongs to.
     */
    Point<dim> reference_position;

    /**
     * Iterator to a cell which owns this atom.
     */
    typename DoFHandler<dim>::active_cell_iterator parent_cell;

  };


}

#endif // __dealii_qc_qc_h
