
#ifndef __dealii_qc_data_out_atom_data_h
#define __dealii_qc_data_out_atom_data_h

#include <deal.II/base/data_out_base.h>

#include <deal.II-qc/atom/cell_molecule_data.h>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A class to write atom data into output streams.
 */
class DataOutAtomData
{
public:
  /**
   * Write out into ostream @p out the atom data contained in
   * @p cell_molcules in the XML vtp format.
   *
   * The function writes out atoms as Points in the vtp format.
   * All the atom attributes are appended to the vtk file as Scalars
   * or Vectors.
   *
   * Each process only writes out the atom data of atoms that are
   * associated to it's locally owned cells. In doing so, each
   * process writes out disjoint sets of atom data.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  void
  write_vtp(const types::CellMoleculeContainerType<dim, atomicity, spacedim>
              &                                  cell_molecules,
            const dealii::DataOutBase::VtkFlags &flags,
            std::ostream &                       out);

  /**
   * Write a pvtp file in order to tell Paraview to group together multiple
   * vtp files that each describe a portion of the data
   * to parallelize visualization.
   */
  void
  write_pvtp_record(const std::vector<std::string> &     vtp_file_names,
                    const dealii::DataOutBase::VtkFlags &flags,
                    std::ostream &                       out);
};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_data_out_atom_data_h */
