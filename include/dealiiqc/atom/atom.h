#ifndef __dealii_qc_atom_h
#define __dealii_qc_atom_h

#include <deal.II/base/point.h>

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
     * Position of the atom in reference coordinates of a cell it belongs to.
     */
    Point<dim> reference_position;

    /**
     * Cell id which owns this atom.
     */
    unsigned int parent_cell;

  };

}

#endif // __dealii_qc_qc_h
