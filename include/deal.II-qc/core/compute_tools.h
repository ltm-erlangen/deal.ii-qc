
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
   * Compute and return the energy and gradient values due to the presence of
   * @p atom with charge @p q (inside domain with @p material_id material id)
   * under external potential fields described through
   * @p external_potential_fields which is a mapping from material ids to
   * PotentialField functions.
   */
  template <int spacedim, bool ComputeGradient=true>
  inline
  std::pair<double, Tensor<1, spacedim> >
  energy_and_gradient
  (const std::multimap<unsigned int, std::shared_ptr<PotentialField<spacedim>>> &external_potential_fields,
   const dealii::types::material_id                                              material_id,
   const Atom<spacedim>                                                         &atom,
   const double                                                                 q)
  {
    // Prepare energy and gradient in the following variables.
    double external_energy = 0.;
    Tensor<1, spacedim> external_gradient;
    external_gradient = 0.;

    // Get the external potential field(s) for the provided material id.
    auto external_potential_fields_range =
      external_potential_fields.equal_range(material_id);

    for (auto
         potential_field  = external_potential_fields_range.first;
         potential_field != external_potential_fields_range.second;
         potential_field++)
      {
        external_energy  += potential_field->second->value (atom.position, q);

        if (ComputeGradient)
          external_gradient +=
            potential_field->second->gradient (atom.position, q);
      }

    return std::make_pair (external_energy, external_gradient);
  }


} // namepsace ComputeTools


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_compute_tools_h */
