
#ifndef __dealii_qc_cluster_weights_by_base_h_
#define __dealii_qc_cluster_weights_by_base_h_

#include <deal.II-qc/atom/cell_molecule_data.h>

#include <deal.II/base/quadrature_lib.h>


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
     * Initialize #sampling_points and #cells_to_sampling_indices data members
     * using @p triangulation and @p quadrature.
     */
    void
    initialize (const Triangulation<dim, spacedim> &triangulation,
                const Quadrature<dim>              &quadrature = QTrapez<dim>());

    /**
     * Return the sampling point with the index given by @p sampling_index.
     */
    inline
    const Point<spacedim> &
    get_sampling_point (const unsigned int sampling_index) const;

    /**
     * Return the number of sampling points.
     */
    inline
    unsigned int
    n_sampling_points () const;

    /**
     * Return the set of sampling indices associated to the @p cell.
     */
    inline
    const std::set<unsigned int> &
    get_sampling_indices (const types::CellIteratorType<dim, spacedim> &cell) const;


    /**
     * Return energy molecules (in a cell based data structure) with
     * appropriately set cluster weights based on provided @p cell_molecules,
     * that were associated to @p mesh, using #cluster_radius and
     * #maximum_cutoff_radius.
     */
    virtual
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const DoFHandler<dim, spacedim>                                  &mesh,
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

  private:

    /**
     * A const pointer to a const Triangulation.
     */
    const Triangulation<dim, spacedim> *tria_ptr;

    /**
     * Map from cells to their corresponding sampling points' indices.
     */
    std::map<types::CellIteratorType<dim,spacedim>, std::set<unsigned int> >
    cells_to_sampling_indices;

    /**
     * A set of global indices of the sampling points that are relevant for
     * the current MPI process.
     */
    std::set<unsigned int> locally_relevant_sampling_indices;
  };

  /*----------------------- Inline functions --------------------------------*/

#ifndef DOXYGEN

  template <int dim, int atomicity, int spacedim>
  const dealii::Point<spacedim> &
  WeightsByBase<dim, atomicity, spacedim>::
  get_sampling_point (const unsigned int sampling_index) const
  {
    Assert (sampling_index < tria_ptr->get_vertices().size(),
            ExcMessage("Invalid sampling index."));
    Assert (locally_relevant_sampling_indices.count(sampling_index),
            ExcMessage("Invalid sampling index. This function was called "
                       "with a sampling index that is not locally relevant."
                       "In other words, the given sampling index is not "
                       "associated to any of the locally relevant cells."));
    return tria_ptr->get_vertices()[sampling_index];
  }



  template <int dim, int atomicity, int spacedim>
  unsigned int
  WeightsByBase<dim, atomicity, spacedim>::
  n_sampling_points () const
  {
    return tria_ptr->get_vertices().size();
  }



  template <int dim, int atomicity, int spacedim>
  const std::set<unsigned int> &
  WeightsByBase<dim, atomicity, spacedim>::
  get_sampling_indices (const types::CellIteratorType<dim, spacedim> &cell) const
  {
    return cells_to_sampling_indices.at(cell);
  }

#endif /* DOXYGEN */

} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_base_h_ */
