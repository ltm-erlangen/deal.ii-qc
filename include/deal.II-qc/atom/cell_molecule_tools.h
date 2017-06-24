
#ifndef __dealii_qc_atom_cell_molecule_tools_h
#define __dealii_qc_atom_cell_molecule_tools_h

#include<deal.II-qc/atom/cell_molecule_data.h>

namespace dealiiqc
{


  /**
   * A namespace for utility functions for CellMoleculeContainerType.
   */
  namespace CellMoleculeTools
  {



    /**
     * Return a pair of range of constant iterators to
     * CellAtomContainerType object @p atoms and the number of atoms of @p atoms
     * in a given @p cell
     *
     * If no molecules are in the queried cell then the function return the
     * pair of (pair of) end iterators and zero. If @p cell_molecules is an
     * empty CellMoleculeContainer, then the function throws an error that
     * the provided CellMoleculeContainer is empty.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    inline
    std::pair
    <
    types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
    ,
    unsigned int
    >
    molecules_range_in_cell (const types::CellIteratorType<dim, spacedim> &cell,
                             const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules)
    {
      AssertThrow (!cell_molecules.empty(),
                   ExcMessage("The given CellMoleculeContainer is empty!"));

      const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
      cell_molecule_range = cell_molecules.equal_range(cell);

      const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      &cell_molecule_range_begin = cell_molecule_range.first,
       &cell_molecule_range_end  = cell_molecule_range.second;

      if (cell_molecule_range_begin == cell_molecule_range_end)
        // Quickly return the following if cell is not
        // found in the CellMoleculeContainerType object
        return std::make_pair(std::make_pair(cell_molecule_range_begin,
                                             cell_molecule_range_end),
                              0);

      // Faster to get the number of molecules in the active cell by
      // computing the distance between first and second iterators
      // instead of calling count on cell_molecules.
      // Here we implicitly cast to unsigned int, but this should be OK as
      // we check that the result is the same as calling count()
      const unsigned int
      n_molecules_in_cell = std::distance (cell_molecule_range.first,
                                           cell_molecule_range.second);

      Assert (n_molecules_in_cell == cell_molecules.count(cell),
              ExcMessage("The number of molecules or energy molecules in the "
                         "cell counted using the distance between the iterator "
                         "ranges yields a different result than "
                         "cell_molecules.count(cell) or"
                         "cell_energy_molecules.count(cell)."));

      return std::make_pair(std::make_pair(cell_molecule_range_begin,
                                           cell_molecule_range_end),
                            n_molecules_in_cell);
    }



    /**
     * Return the number of molecules in @p cell_energy_molecules, associated to @p cell,
     * which have non-zero cluster weights.
     */
    template<int dim, int atomicity=1, int spacedim=dim>
    inline
    unsigned int
    n_cluster_molecules_in_cell (const types::CellIteratorType<dim, spacedim> &cell,
                                 const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_energy_molecules)
    {
      const types::CellMoleculeConstIteratorRangeType<dim, atomicity, spacedim>
      cell_molecule_range = cell_energy_molecules.equal_range(cell);

      const types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      &cell_molecule_range_begin = cell_molecule_range.first,
       &cell_molecule_range_end  = cell_molecule_range.second;

      if (cell_molecule_range_begin == cell_molecule_range_end)
        // Quickly return the following if cell is not
        // found in the CellMoleculeContainerType object
        return 0;

      unsigned int n_cluster_molecules_in_this_cell = 0;

      for (types::CellMoleculeConstIteratorType<dim>
           cell_molecule_iterator  = cell_molecule_range_begin;
           cell_molecule_iterator != cell_molecule_range_end;
           cell_molecule_iterator++)
        {
          Assert (cell_molecule_iterator->second.cluster_weight != numbers::invalid_cluster_weight,
                  ExcMessage("At least one of the molecule's cluster weight is "
                             "not initialized to a valid number."
                             "This function should be called only after "
                             "setting up correct cluster weights."));
          if (cell_molecule_iterator->second.cluster_weight != 0)
            n_cluster_molecules_in_this_cell++;
        }


      return n_cluster_molecules_in_this_cell;
    }



  } // namespace CellMoleculeTools


} // namespace dealiiqc

#endif /* __dealii_qc_atom_cell_molecule_tools_h */
