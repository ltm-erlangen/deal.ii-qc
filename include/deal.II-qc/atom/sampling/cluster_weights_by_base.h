
#ifndef __dealii_qc_cluster_weights_by_base_h_
#define __dealii_qc_cluster_weights_by_base_h_

#include <deal.II-qc/atom/cell_molecule_data.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{

  /**
   * Base class for assigning @see cluster_weight to energy molecules.
   *
   * The spheres with radii #cluster_radius around the sampling points are
   * called clusters. Within each cell of the @p mesh, clusters are formed
   * around quadrature or sampling points to identify energy molecules.
   * A molecule is an energy molecule if its initial location is
   * within a #cluster_radius plus #maximum_cutoff_radius distance to a
   * sampling points. An energy molecule is a cluster or sampling molecule
   * if it is within a #cluster_radius distance to a sampling point.
   *
   * The returned energy molecules may have non-zero (cluster molecules) or
   * zero (non-cluster molecules) cluster weights.
   *
   * The terms sampling molecules and cluster molecules are used
   * interchangeably; also the terms sampling points and quadrature points
   * in the context of sampling rules are used interchangeably.
   */
  template <int dim, int atomicity=1, int spacedim=dim>
  class WeightsByBase
  {
  public:

    // TODO: Add sampling points around which clusters are to be formed
    //       (points around where sampling molecules are formed).
    /**
     * Constructor.
     */
    WeightsByBase (const double &cluster_radius,
                   const double &maximum_cutoff_radius);


    virtual ~WeightsByBase();

    /**
     * Return energy molecules (in a cell based data structure) with
     * appropriately set cluster weights based on provided @p cell_molecules,
     * that were associated to @p mesh, using #cluster_radius and
     * #maximum_cutoff_radius.
     */
    virtual
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const dealii::DoFHandler<dim, spacedim>                          &mesh,
     const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const = 0;

  protected:

    /**
     * The cluster radius for the QC approach.
     */
    const double cluster_radius;

    /**
     * The maximum of cutoff radii.
     */
    const double maximum_cutoff_radius;

    // TODO: sampling_points
  };


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_base_h_ */
