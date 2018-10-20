
#ifndef __dealii_qc_cluster_weights_by_lumped_vertex_h_
#define __dealii_qc_cluster_weights_by_lumped_vertex_h_

#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{
  /**
   * A class which creates a cluster around each sampling point and calculates
   * cluster weights using lumping procedure based on
   * <a href="https://doi.org/10.1016/S0022-5096(01)00034-5">An analysis of
   * the quasicontinuum method</a> by Knap et al explained below.
   *
   * Let \f$\mathcal{C}\f$ be the set of all clusters with cardinality
   * \f$N^c\f$ for an atomistic system with the set of total lattice sites
   * \f$\mathscr{L}\f$. If a sampling molecule \f$i\f$ is in a cluster \f$I\f$
   * then \f$i \in I \in \mathcal{C} \subset \mathscr{L}\f$.
   * The cluster weights \f$w^{}_I\f$ are a function of cluster radius.
   * For a given cluster radius, the cluster weights are obtained by
   * requiring that the shape functions are summed exactly. In other words,
   * the weighted sum of all the shape function values at the position of
   * sampling molecules must be exactly equal to the sum of shape function
   * values at all the lattice sites. The application of the summation rule to
   * shape functions leads to the system of \f$N^c\f$ linear algebraic
   * equations,
   * \f[
   *     \sum_{J \, \in \, \mathcal{C} }
   *     A^{}_{IJ} w^{}_I =
   *     b_I, \,\, \forall I \, \in \, \mathcal{C},
   * \f]
   * where the matrix \f$A\f$ is defined as
   * \f[
   *     A^{}_{IJ} = \sum_{l \, \in \, J} \, N^{}_I ( \textbf{X}_l),
   *     \,\, \forall \,\, I \mbox{ and } J \, \in \, \mathcal{C}
   * \f]
   * and
   * \f[
   *     b^{}_{I} = \sum_{l \, \in \, \mathscr{L}} \, N^{}_I (\textbf{X}_l),
   *     \,\, \forall \,\, I  \, \in \, \mathcal{C}.
   * \f]
   *
   * Note that each entry of the matrix \f$A\f$ is due to a pair of clusters
   * \f$I\f$ and \f$J\f$. The calculation of the array \f$b\f$ requires the
   * evaluation of the shape functions at all lattice sites within a crystal.
   * While this may be regarded as acceptable for small crystals, it becomes
   * prohibitively expensive when applied to large samples.
   *
   * A simpler alternative route proposed to avoid solving the possibly large
   * system of equation is to resort to lumping in order to replace \f$A\f$
   * by a diagonal matrix. The diagonal entries of the lumped matrix are
   * \f[
   *     A^{}_{II} = \sum_{J \, \in \, \mathcal{C}} \,\,
   *                 \sum_{ l \, \in \, J}
   *                 N_{I} (\textbf{X}_l), \,\,
   *                 \forall I \, \in \, \mathcal{C}
   * \f]
   * with all other entries in \f$A\f$ set to zero. Once the matrix \f$A\f$ is
   * lumped, the cluster weight of cluster \f$I\f$ can be calculated as
   * \f[
   *     w_{I} = \frac{b_I}{A_{II}}, \,\,\, \forall
   *     I \, \in \, \mathcal{C}.
   * \f]
   */
  template <int dim, int atomicity = 1, int spacedim = dim>
  class WeightsByLumpedVertex : public WeightsByBase<dim, atomicity, spacedim>
  {
  public:
    /**
     * Constructor.
     */
    WeightsByLumpedVertex(const double &cluster_radius,
                          const double &maximum_energy_radius);

    /**
     * @see WeightsByBase::update_cluster_weights() and WeightsByLumpedCell
     * class description.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights(
      const Triangulation<dim, spacedim> &triangulation,
      const types::CellMoleculeContainerType<dim, atomicity, spacedim>
        &cell_molecules) const;
  };


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_cluster_weights_by_lumped_vertex_h_ */
