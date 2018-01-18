
#ifndef __dealii_qc_weights_by_optimal_summation_rules_h_
#define __dealii_qc_weights_by_optimal_summation_rules_h_

#include <deal.II-qc/atom/cell_molecule_data.h>

#include <deal.II/base/quadrature_lib.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{

  // TODO: Implementation (first order summation rule for now).
  /**
   * The summation rules can be identified by two numbers: \f$ n_s\f$ the number
   * of shells included in clusters around the cluster-type sampling atom, and
   * \f$n_q\f$, the number of quadrature-type sampling atoms per element
   * (clusters around these sampling atoms are not considered).
   * Hence, a summation rule is characterized by the pair
   * (\f$n_s\f$, \f$n_q\f$). For example in two-dimensions (0,1) summation rule
   * (which is classified as to a first-order optimal summation rule)
   * indicates that a quadrature-type sampling atom (if exists) at the center
   * of the element will also contribute towards energy and force computations.
   * Such a sampling atom is also termed as inner-element sampling atom.
   * The inner-element sampling atom represents all inner-element lattice sites
   * except those within the representative distance (\f$r_{rep}\f$) of
   * cluster-type sampling atoms.
   *
   * Nomenclature of summation rules:
   * - (0,0): a purely nodal summation rule,
   * - (\f$n_s\f$, 0): nodal cluster summation rules with \f$n_s\f$ shells of
   * cluster atoms,
   * - (0, \f$n_q\f$): quadrature summation rules with \f$n_q\f$ quadrature-type
   * sampling atoms per element,
   * - (0, 1): a first-order optimal summation rule with an inner-element
   * sampling atom.
   * - (0, 1, 4): a second-order optimal summation rule with an inner-element
   * sampling atom along with four face center sampling atoms in the case of
   * quadrilateral element ((0, 1, 6) for hexagonal elements).
   *
   * The representative distance (\f$r_{rep}\f$)(required to determine the
   * weights of sampling atoms) is to be obtained through numerical
   * experimentation. It is therefore possible that the representative distance
   * is not a unique constant for a chosen interaction potential for
   * a material for any arbitrary simulation scenarios.
   *
   * Also see <a href="https://doi.org/10.1016/j.jmps.2015.03.007">
   * Summation rules for a fully nonlocal energy-based quasicontinuum method</a>
   * by Amelang, Venturini and Kochmann.
   */
  template <int dim, int atomicity=1, int spacedim=dim>
  class WeightsByOptimalSummationRules
  {

    // TODO: More documentation.
    /**
     * Constructor.
     */
    WeightsByOptimalSummationRules (const double &cluster_radius,
                                    const double &maximum_energy_radius);

    // TODO: More documentation.
    /**
     * @see WeightsByBase::update_cluster_weights().
     *
     * The approach of WeightsByOptimalSummationRules ...
     */
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    update_cluster_weights
    (const Triangulation<dim, spacedim>                               &triangulation,
     const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const;

  };

}


DEAL_II_QC_NAMESPACE_CLOSE

#endif /* __dealii_qc_weights_by_optimal_summation_rules_h_ */
