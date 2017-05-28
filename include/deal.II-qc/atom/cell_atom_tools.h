
#ifndef __dealii_qc_atom_cell_atom_tools_h
#define __dealii_qc_atom_cell_atom_tools_h

#include<deal.II-qc/atom/atom_data.h>

namespace dealiiqc
{


  /**
   * A namespace for utility functions for CellAtomContainerType.
   */
  namespace CellAtomTools
  {



    /**
     * Return a pair of range of constant iterators to
     * CellAtomContainerType object @p atoms and the number of atoms of @p atoms
     * in a given @p cell
     *
     * If no atoms are in the queried cell then the function return the pair of
     * (pair of) end iterators and zero. If an empty CellAtomContainer is
     * passed, then the function throws an error that the CellAtomContainer is
     * empty.
     */
    template<int dim>
    inline
    std::pair<types::CellAtomConstIteratorRangeType<dim>, unsigned int>
    atoms_range_in_cell (const types::CellIteratorType<dim> &cell,
                         const types::CellAtomContainerType<dim> &atoms)
    {
      AssertThrow (!atoms.empty(),
                   ExcMessage("The given CellAtomContainer is empty!"));

      const types::CellAtomConstIteratorRangeType<dim>
      cell_atom_range = atoms.equal_range(cell);

      const types::CellAtomConstIteratorType<dim>
      &cell_atom_range_begin = cell_atom_range.first,
       &cell_atom_range_end  = cell_atom_range.second;

      if (cell_atom_range_begin == cell_atom_range_end)
        // Quickly return the following if cell is not
        // found in the CellAtomContainerType object
        return std::make_pair(std::make_pair(cell_atom_range_begin,
                                             cell_atom_range_end),
                              0);

      // Faster to get the number of atoms in the active cell by
      // computing the distance between first and second iterators
      // instead of calling count on energy_atoms.
      // Here we implicitly cast to usngined int, but this should be OK as
      // we check that the result is the same as calling count()
      const unsigned int
      n_atoms_in_cell = std::distance (cell_atom_range.first,
                                       cell_atom_range.second);

      Assert (n_atoms_in_cell == atoms.count(cell),
              ExcMessage("The number of atoms or energy atoms in the cell "
                         "counted using the distance between the iterator "
                         "ranges yields a different result than "
                         "atoms.count(cell) or"
                         "energy_atoms.count(cell)."));

      return std::make_pair(std::make_pair(cell_atom_range_begin,
                                           cell_atom_range_end),
                            n_atoms_in_cell);
    }



    /**
     * Return the number of atoms in @p energy_atoms, associated to @p cell,
     * which have non-zero cluster weights.
     */
    template<int dim>
    inline
    unsigned int
    n_cluster_atoms_in_cell (const types::CellIteratorType<dim> &cell,
                             const types::CellAtomContainerType<dim> &energy_atoms)
    {
      const types::CellAtomConstIteratorRangeType<dim>
      cell_atom_range = energy_atoms.equal_range(cell);

      const types::CellAtomConstIteratorType<dim>
      &cell_atom_range_begin = cell_atom_range.first,
       &cell_atom_range_end  = cell_atom_range.second;

      if (cell_atom_range_begin == cell_atom_range_end)
        // Quickly return the following if cell is not
        // found in the CellAtomContainerType object
        return 0;

      unsigned int n_cluster_atoms_in_this_cell = 0;

      for (types::CellAtomConstIteratorType<dim>
           cell_atom_iterator  = cell_atom_range_begin;
           cell_atom_iterator != cell_atom_range_end;
           cell_atom_iterator++)
        {
          Assert (cell_atom_iterator->second.cluster_weight != number::invalid_cluster_weight,
                  ExcMessage("At least one of the atom's cluster weight is "
                             "not initialized to a valid number."
                             "This function should be called only after "
                             "setting up correct cluster weights."));
          if (cell_atom_iterator->second.cluster_weight != 0)
            n_cluster_atoms_in_this_cell++;
        }


      return n_cluster_atoms_in_this_cell;
    }



  } // namespace CellAtomTools


} // namespace dealiiqc

#endif /* __dealii_qc_atom_cell_atom_tools_h */
