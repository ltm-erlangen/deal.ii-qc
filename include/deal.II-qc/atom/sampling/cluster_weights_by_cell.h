
#ifndef __dealii_qc_cluster_weights_by_cell_h_
#define __dealii_qc_cluster_weights_by_cell_h_

#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>

namespace dealiiqc
{

  namespace Cluster
  {

    /**
     * A derived class for updating cluster weights using the cell approach.
     */
    template <int dim>
    class WeightsByCell : public WeightsByBase <dim>
    {
    public:

      /**
       * Constructor
       */
      WeightsByCell (const double &cluster_radius,
                     const double &maximum_energy_radius);

      /**
       * @copydoc WeightsByBase::update_cluster_weights()
       *
       * The approach of WeightsByCell counts the number of non-energy atoms
       * (i.e., the atoms in locally relevant cells but are not close enough
       * to cell's vertices that they are not energy atoms) which is then used
       * to compute cluster weights of the cluster atoms.
       */
      types::CellAtomContainerType<dim>
      update_cluster_weights (const types::MeshType<dim> &mesh,
                              const types::CellAtomContainerType<dim> &atoms) const;

    };


  } // namespace Cluster


} // namespace dealiiqc



#endif /* __dealii_qc_cluster_weights_by_cell_h_ */
