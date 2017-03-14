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
  class Atom
  {
  public:
    /**
     * Default constructor.
     */
    Atom();

    /**
     * Position in real space
     */
    Point<dim> position;

    /**
     * Charge of the atom
     */
    types::charge q;

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
