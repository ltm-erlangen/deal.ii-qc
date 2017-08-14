
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
     * Initialize #locally_relevant_sampling_indices and
     * #cells_to_sampling_indices data members using @p triangulation and
     * @p quadrature.
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
    const std::vector<unsigned int> &
    get_sampling_indices (const types::CellIteratorType<dim, spacedim> &cell) const;

    /**
     * Return the set of sampling points associated to the @p cell.
     */
    inline
    std::vector<Point<spacedim> >
    get_sampling_points (const types::CellIteratorType<dim, spacedim> &cell) const;

    /**
     * Return energy molecules (in a cell based data structure) with
     * appropriately set cluster weights based on provided @p cell_molecules,
     * that were associated to @p triangulation, using #cluster_radius and
     * #maximum_cutoff_radius.
     */
    virtual
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const Triangulation<dim, spacedim>                               &triangulation,
     const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const = 0;

    /**
     * Prepare the inverse masses attributed to the locally relevant dofs
     * in @p inverse_masses, given @p dof_handler and cell based molecule data
     * @p cell_molecule_data of the atomistic system.
     * It is assumed that energy molecules of @p cell_molecule_data have
     * updated cluster weights.
     *
     * For each degree of freedom associated to a sampling point in conjunction
     * with an atom stamp, the mass attributed to the dof is computed by
     * summing up the masses of sampling molecules' atoms with the same
     * atom stamp, scaled by their respective molecules' cluster weights.
     * Mathematically, if \f$ u^i_d \f$ is a degree of freedom associated to
     * the sampling point \f$ p \f$ in conjunction with atom stamp \f$ i \f$,
     * then the mass \f$ m^i_d\f$ attributed to this dof is given as
     * \f[
     *     m^i_d  = \sum_{I \, \in \, \cal C_p}    w^{}_I \, \mathsf m^i_I,
     * \f]
     * where \f$ \cal C_p \f$ is the set of sampling molecules associated to the
     * sampling point \f$ p \f$ and \f$ \mathsf m^i_I \f$ is the mass of the
     * atom with atom stamp \f$ i \f$ of molecule \f$ I \f$.
     *
     * @note The @p inverse_masses vector should be a writable ghosted vector
     * that is prepared for example in the following way:
     * @code
     *   TrilinosWrappers::MPI::Vector inverse_masses;
     *   inverse_masses.reinit (dof_handler.locally_owned_dofs(),
     *                          locally_relevant_set,
     *                          mpi_communicator,
     *                          true);
     * @endcode
     */
    template <typename VectorType>
    void
    compute_dof_inverse_masses
    (VectorType                                       &inverse_masses,
     const DoFHandler<dim, spacedim>                  &dof_handler,
     const CellMoleculeData<dim, atomicity, spacedim> &cell_molecule_data) const;

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

    // TODO: Remove this member when the chosen Quadrature<> is other than
    //       QTrapez<>.
    /**
     * A const pointer to a const Triangulation.
     */
    const Triangulation<dim, spacedim> *tria_ptr;

    /**
     * Map from cells to their corresponding sampling points' indices.
     */
    std::map<types::CellIteratorType<dim,spacedim>, std::vector<unsigned int> >
    cells_to_sampling_indices;

    /**
     * A set of global indices of the sampling points that are relevant for
     * the current MPI process.
     */
    IndexSet locally_relevant_sampling_indices;
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
    Assert (locally_relevant_sampling_indices.is_element(sampling_index),
            ExcMessage("Invalid sampling index. This function was called "
                       "with a sampling index that is not locally relevant."
                       "In other words, the given sampling index is not "
                       "associated to any of the locally relevant cells."));

    // TODO: Change when chosen Quadrature<> is other than QTrapez<>.
    return tria_ptr->get_vertices()[sampling_index];
  }



  template <int dim, int atomicity, int spacedim>
  unsigned int
  WeightsByBase<dim, atomicity, spacedim>::
  n_sampling_points () const
  {
    // TODO: Change when chosen Quadrature<> is other than QTrapez<>.
    return tria_ptr->get_vertices().size();
  }



  template <int dim, int atomicity, int spacedim>
  const std::vector<unsigned int> &
  WeightsByBase<dim, atomicity, spacedim>::
  get_sampling_indices (const types::CellIteratorType<dim, spacedim> &cell) const
  {
    return cells_to_sampling_indices.at(cell);
  }



  template <int dim, int atomicity, int spacedim>
  std::vector<Point<spacedim> >
  WeightsByBase<dim, atomicity, spacedim>::
  get_sampling_points (const types::CellIteratorType<dim, spacedim> &cell) const
  {
    // Get the global indices of the sampling points of this cell.
    const std::vector<unsigned int> &this_cell_sampling_indices =
      this->get_sampling_indices(cell);

    // Prepare sampling points of this cell in this container.
    std::vector<Point<spacedim> > this_cell_sampling_points;

    this_cell_sampling_points.reserve(this_cell_sampling_indices.size());

    // TODO: remove get_sampling_point.
    // Prepare sampling points of this cell.
    for (const auto &sampling_index : this_cell_sampling_indices)
      this_cell_sampling_points.push_back(this->get_sampling_point(sampling_index));

    return this_cell_sampling_points;
  }

#endif /* DOXYGEN */

} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_base_h_ */
