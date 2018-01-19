
#ifndef __dealii_qc_weights_by_optimal_summation_rules_h_
#define __dealii_qc_weights_by_optimal_summation_rules_h_

#include <deal.II-qc/atom/cell_molecule_data.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{

  // TODO: Implementation (first order summation rule for now).
  /**
   * Summation rules can be identified by two numbers: \f$ n_s\f$ the number
   * of shells included in clusters around the (vertex-type) sampling
   * atom/molecule, and \f$n_q\f$, the number of sampling atoms/molecules
   * (per element) that are either inside or on the face or edges of the element
   * (clusters around these sampling atoms/molecules are not considered).
   * Hence, a summation rule is characterized by the pair
   * (\f$n_s\f$, \f$n_q\f$). For example in two-dimensions (0,1) summation rule
   * (which is classified as to a first-order optimal summation rule)
   * indicates that a sampling atom/molecule (if exists) at the center
   * of the element will also contribute towards energy and force computations.
   * Such a sampling atom/molecule is also termed as inner-element sampling
   * atom/molecule. The inner-element sampling atom/molecule represents
   * all inner-element lattice sites except those within the representative
   * distance (\f$r_{rep}\f$) of vertex-type sampling atoms/molecules.
   *
   * Nomenclature of summation rules:
   * - (0,0): a purely nodal summation rule,
   * - (\f$n_s\f$, 0): nodal cluster summation rules with \f$n_s\f$ shells of
   * cluster atoms/molecules,
   * - (0, \f$n_q\f$): quadrature summation rules with \f$n_q\f$ quadrature-type
   * sampling atoms per element,
   * - (0, 1): a first-order optimal summation rule with an inner-element
   * sampling atom.
   * - (0, 1, 4): a second-order optimal summation rule with an inner-element
   * sampling atom/molecule along with four face center sampling atoms/molecules
   * in the case of quadrilateral element ((0, 1, 6) for hexagonal elements).
   *
   * The representative distance (\f$r_{rep}\f$)(required to determine the
   * weights of sampling atoms/molecules) is to be obtained through numerical
   * experimentation. It is therefore possible that the representative distance
   * is not a unique constant for a chosen interaction potential for
   * a material for any arbitrary simulation scenarios.
   *
   * Also see <a href="https://doi.org/10.1016/j.jmps.2015.03.007">
   * Summation rules for a fully nonlocal energy-based quasicontinuum method</a>
   * by Amelang, Venturini and Kochmann.
   */
  template <int dim, int atomicity=1, int spacedim=dim>
  class WeightsByOptimalSummationRules : public WeightsByBase<dim, atomicity, spacedim>
  {
  public:

    // TODO: More documentation.
    /**
     * Constructor.
     */
    WeightsByOptimalSummationRules (const double &cluster_radius,
                                    const double &maximum_energy_radius,
                                    const double &rep_distance);

    // TODO: More documentation & implementation.
    /**
     * @see WeightsByBase::update_cluster_weights().
     *
     * The approach of WeightsByOptimalSummationRules requires that a variable
     * called representative distance is set based on numerical experimentation.
     *
     * Three specific cases exist:
     * - In the fully atomistic region, cells do not contain sampling
     *   atoms/molecules other than that of vertex-type.
     *   In such a case, all the (vertex-type)
     *   sampling points of the cell are given sampling weight of 1.
     * - In the interface region (which typically come with hanging nodes),
     *   the representative spheres of (vertex-type) sampling atoms/molecules
     *   could potentially overlap. The representative spheres should be split to
     *   determine sampling weights.
     * - In the continuum region, cells are large enough that none of the
     *   representative spheres overlap, the vertex-type sampling
     *   atoms/molecules get a sampling weight of
     *   \f$ 4/3 \pi r_r^3 \rho\f$ (for three-dimensions)
     *   or \f$ \pi r_r^2 \rho\f$ (for two-dimensions),
     *   where \f$ \rho\f$ is the atom/molecule number density of
     *   the atomistic system and \f$r_r\f$ is the representative distance of
     *   the vertex-type sampling atoms/molecules.
     *
     * For the first order summation rules, the sampling weight of
     * the inner-element sampling atom/molecule (if exists), is given as
     * \f$ w_q = \rho \nu - \sum_i{w_i}\f$
     * where \f$ \rho\f$ is the atom/molecule number density of the atomistic
     * system, \f$ w_i\f$ is the sampling weight of the vertex-type
     * sampling atom \f$i\f$ of the cell and \f$ \nu\f$ is the volume of the
     * cell containing the quadrature-type sampling atom.
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const Triangulation<dim, spacedim>                               &triangulation,
     const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const;

  protected:

    /**
     * Representative distance of the vertex-type sampling atoms/molecules.
     */
    double rep_distance;

  };

}


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_weights_by_optimal_summation_rules_h_ */
