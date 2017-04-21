

#ifndef __dealii_qc_atom_data_h
#define __dealii_qc_atom_data_h

#include <dealiiqc/atom/atom.h>

namespace dealiiqc
{

  namespace types
  {
    /**
     * A typedef for mesh.
     */
    template<int dim>
    using MeshType = dealii::DoFHandler<dim>;

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    template<int dim>
    using CellIteratorType = typename MeshType<dim>::active_cell_iterator;

    /**
     * A typedef for container that holds cell and associated atoms
     */
    template<int dim>
    using CellAtomContainerType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >;

    /**
     * A typedef for iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >::iterator;

    /**
     * A typedef for const_iterator over CellAtomContainerType
     */
    template<int dim>
    using CellAtomIteratorType = typename std::multimap< CellIteratorType<dim>, Atom<dim> >::iterator;

  } // types

  //TODO: Create an AtomData struct here


}


#endif /* __dealii_qc_atom_data_h */
