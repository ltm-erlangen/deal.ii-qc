#ifndef __dealii_qc_atom_h
#define __dealii_qc_atom_h

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <dealiiqc/utility.h>

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


  /**
   * A namespace for auxiliary functions that are
   */
  namespace atom
  {

    /**
     * Function to check if the atom is a within a certain distance
     * @p distance from the vertices of it's parent cell.
     *
     * If the parent cell is not initialized
     * the function throws an exception.
     *
     * This function is not included in the Atom struct to
     * make the Atom struct lighter.
     */
    template<int dim>
    inline
    bool
    is_within_distance_from_vertices( const Atom<dim> &atom, const double &distance)
    {
      // Throw exception if the parent_cell is not set or is not in a valid
      // cell iterator state.
      AssertThrow( atom.parent_cell->state() == IteratorState::valid,
                   ExcMessage( "Either parent_cell of the atom is not initialized or"
                               "the parent_cell iterator points past the end"));

      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        if (  (atom.parent_cell->vertex(v)- atom.position).norm_square()
              < dealii::Utilities::fixed_power<2>( distance ) )
          return true;
      return false;
    }
  } // Atoms namespace

}

#endif // __dealii_qc_qc_h
