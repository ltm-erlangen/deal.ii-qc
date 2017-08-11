
#ifndef __dealii_qc_compute_tools_h
#define __dealii_qc_compute_tools_h

#include <deal.II-qc/atom/molecule.h>
#include <deal.II-qc/potentials/potential_field.h>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A namespace for tools to compute energy and gradient among atoms
 * and molecules.
 */
namespace ComputeTools
{

  /**
   * Compute and return the energy and gradient values due to the presense of
   * @p atom with charge @p q under a potential field described by the scalar
   * valued function @p external_potential_field.
   */
  template <int spacedim, bool ComputeGradient=true>
  inline
  std::pair<double, Tensor<1, spacedim> >
  energy_and_gradient
  (const PotentialField<spacedim> &external_potential_field,
   const Atom<spacedim>           &atom,
   const double                    q)
  {
    const Tensor<1, spacedim> external_gradient =
      ComputeGradient                                      ?
      external_potential_field.gradient (atom.position, q) :
      Tensor<1, spacedim>();

    return std::make_pair (external_potential_field.value (atom.position, q),
                           external_gradient);
  }


} // namepsace ComputeTools


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_compute_tools_h */
