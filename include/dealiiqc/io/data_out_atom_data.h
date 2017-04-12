
#ifndef __dealii_qc_data_out_atom_data_h
#define __dealii_qc_data_out_atom_data_h

#include <deal.II/base/data_out_base.h>

#include <dealiiqc/atom/atom_handler.h>


namespace dealiiqc
{
  using namespace dealii;

  /**
   * A class to write atom data into output streams.
   */
  template< int dim>
  class DataOutAtomData
  {
  public:

    /**
     * A typedef for container that carries
     * cell and atom association information
     */
    using CellAtomContainerType = typename std::multimap< typename DoFHandler<dim>::active_cell_iterator, Atom<dim> >;

    /**
     * Write out into ostream @p out the atom data contained in
     * @p cell_atom_container in the XML vtp format.
     *
     * The function writes out atoms as Points in the vtp format.
     * All the atom attributes are appended to the vtk file as Scalars
     * or Vectors.
     *
     * Each process only writes out the atom data of atoms that are
     * associated to it's locally owned cells. In doing so, each
     * process writes out disjoint sets of atom data.
     */
    void write_vtp ( const CellAtomContainerType &cell_atom_container,
                     const dealii::DataOutBase::VtkFlags &flags,
                     std::ostream &out);

    /**
     * Write a pvtp file in order to tell Paraview to group together multiple
     * vtp files that each describe a portion of the data
     * to parallelize visualization.
     */
    void write_pvtp_record( const std::vector<std::string> &vtp_file_names,
                            const dealii::DataOutBase::VtkFlags &flags,
                            std::ostream &out);

  };

}


#endif /* __dealii_qc_data_out_atom_data_h */
