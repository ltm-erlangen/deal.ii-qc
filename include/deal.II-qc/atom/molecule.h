#ifndef __dealii_qc_molecule_h
#define __dealii_qc_molecule_h

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II-qc/atom/atom.h>

namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class for molecules embedded in a <tt>spacedim</tt>-dimensional space.
   */
  template<int stamps, int spacedim>
  struct Molecule
  {

    /**
     * A list of atoms that constitute this molecule. Each atom in the molecule
     * is given a different stamp and therefore the order of atoms is important
     * and is according to their stamp.
     */
    std::array<Atom<spacedim>, stamps> atoms;

    /**
     * Position of the molecule in reference coordinates of the cell to which
     * it is associated to.
     */
    Point<spacedim> position_inside_reference_cell;

    /**
     * Representativeness of this molecule in its contribution to the
     * total energy of the atomistic system.
     *
     * A molecule is inside a cluster if any of its atoms are geometrically
     * inside the cluster. Any molecule that is located inside a cluster is a
     * cluster molecule or sampling molecule. All the sampling molecules have
     * non-zero @p cluster_weight.
     */
    double cluster_weight;

  };


} // namespace dealiiqc



#endif // __dealii_qc_molecule_h
