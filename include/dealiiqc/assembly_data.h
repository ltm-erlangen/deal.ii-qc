
#ifndef __dealii_qc_cell_data_h
#define __dealii_qc_cell_data_h

#include <dealiiqc/atom/atom_data.h>

namespace dealiiqc
{

  // TODO: Update this struct after finalizing force computing algorithm.
  /**
   * Auxiliary class with all the information needed per cell for
   * calculation of energy and forces in quasi-continuum method.
   *
   * Since initial positions of atoms is generally random in each
   * element, we have to have a separate FEValues object for each cell.
   *
   * The most tricky part in non-local methods like molecular mechanics
   * within the FE approach is to get the following association link:
   *
   * cell -> atom_id -> neighbour_id -> neighbour_cell -> local_neighbour_id
   *
   */
  struct AssemblyData
  {
    AssemblyData()
    {
    };

    ~AssemblyData()
    {
      Assert(fe_values.use_count() < 2,
             ExcMessage("use count: " + std::to_string(fe_values.use_count())));
    }

    /**
     * FEValues object to evaluate fields and shape function values at
     * quadrature points.
     */
    std::shared_ptr<FEValues<dim>> fe_values;

    /**
     * All atoms attributed to this cell.
     *
     * TOOD: move away from this struct? Do-once-and-forget.
     */
    std::vector<unsigned int> cell_atoms;

    /**
     * Any atom contributing to (QC) energy of the system is in @see energy_atoms.
     * An atom contributes to (QC) energy computations if it happens to be
     * located within a distance of @see cluster_radius + @see cutoff_radius
     * to any vertex.
     */
    //std::multimap< CellIterator, Atom<dim>> energy_atoms;

    /**
     * A map of global atom IDs to quadrature point (local id of at atom)
     */
    std::map<unsigned int, unsigned int> quadrature_atoms;

    /**
     * IDs of all atoms which are needed for energy calculation.
     */
    std::vector<unsigned int> energy_atoms;

    /**
     * A vector to store displacements evaluated at quadrature points
     */
    mutable std::vector<Tensor<1,dim>> displacements;

    /**
     * A map for each cell to related global degree-of-freedom, to those
     * defined on the cell. Essentially, the reverse of
     * cell->get_dof_indices().
     */
    std::map<unsigned int, unsigned int> global_to_local_dof;

  };

  namespace types
  {

    /**
     * A typedef for map of cells to data.
     */
    template<int dim>
    using CellAssemblyData = typename std::map< types::CellIteratorType<dim>, AssemblyData>;

  } // types

}



#endif /* __dealii_qc_cell_data_h */
