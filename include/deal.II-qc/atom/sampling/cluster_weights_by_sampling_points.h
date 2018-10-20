
#ifndef __dealii_qc_cluster_weights_by_sampling_points_h_
#define __dealii_qc_cluster_weights_by_sampling_points_h_

#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{
  /**
   * A derived class for updating cluster weights using the sampling points
   * approach, in other words using Voronoi tessellation of the cluster's
   * sampling points.
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  class WeightsBySamplingPoints : public WeightsByBase<dim, atomicity, spacedim>
  {
  public:
    /**
     * Constructor.
     */
    WeightsBySamplingPoints(const double &cluster_radius,
                            const double &maximum_energy_radius);

    /**
     * @see WeightsByBase::update_cluster_weights().
     *
     * The approach of WeightsBySamplingPoints counts the total number of
     * molecules associated to each sampling point and the number of molecules
     * within the cluster of the sampling points; their ratio is used as
     * cluster weights. The former of the two numbers is nothing else but
     * the number of molecules in Voronoi cell associated to a given sampling
     * point.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights(
      const Triangulation<dim, spacedim> &triangulation,
      const types::CellMoleculeContainerType<dim, atomicity, spacedim>
        &cell_molecules) const;
  };

} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_sampling_points_h_ */
