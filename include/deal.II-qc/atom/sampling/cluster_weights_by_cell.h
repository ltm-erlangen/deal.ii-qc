
#ifndef __dealii_qc_cluster_weights_by_cell_h_
#define __dealii_qc_cluster_weights_by_cell_h_

#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{

  /**
   * A derived class for updating cluster weights using the cell approach.
   */
  template <int dim, int atomicity=1, int spacedim=dim>
  class WeightsByCell : public WeightsByBase <dim, atomicity, spacedim>
  {
  public:

    /**
     * Constructor
     */
    WeightsByCell (const double &cluster_radius,
                   const double &maximum_energy_radius);

    /**
     * @see WeightsByBase::update_cluster_weights().
     *
     * The approach of WeightsByCell counts the total number of molecules and
     * the number of cluster molecules within each cell; their ratio is used
     * as cluster weights.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const dealii::DoFHandler<dim, spacedim>                          &mesh,
     const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const;

  };


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_cell_h_ */
