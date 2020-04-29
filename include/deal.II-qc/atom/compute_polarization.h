#ifndef __dealii_qc_compute_polarization_h
#define __dealii_qc_compute_polarization_h

#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II-qc/utilities.h>


/**
 * A post processor to compute polarization, a single vector quantity
 * (defined as having exactly @tparam dim components), from
 * the finite element displacement field passed to the DataOut class.
 */
template <int dim, int atomicity>
class ComputePolarization : public dealii::DataPostprocessorVector<dim>
{
public:
  /**
   * Constructor.
   * Take the list of atomicity number of charges corresponding to the atoms
   * of a representative molecule in the simultion.
   */
  ComputePolarization(
    const std::array<dealiiqc::types::charge, atomicity> &charges,
    const std::string solution_name = "Polarization");

  /**
   * This is the central function performing the polarization computations.
   * The second argument @p computed_quantities is a reference to
   * the postprocessed polarization data which already has the correct size.
   * The function takes the values of the solution (displacement field)
   * @p inputs at all evalutation points within a cell.
   */
  virtual void
  evaluate_vector_field(
    const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<dealii::Vector<double>> &computed_quantities) const override;

private:
  /**
   * A list of atomicity number of charges.
   */
  const std::array<dealiiqc::types::charge, atomicity> atomicity_charges;
};



template <int dim, int atomicity>
ComputePolarization<dim, atomicity>::ComputePolarization(
  const std::array<dealiiqc::types::charge, atomicity> &charges,
  const std::string                                     solution_name)
  : dealii::DataPostprocessorVector<dim>(solution_name,
                                         dealii::UpdateFlags::update_values)
  , atomicity_charges(charges)
{}



template <int dim, int atomicity>
void
ComputePolarization<dim, atomicity>::evaluate_vector_field(
  const dealii::DataPostprocessorInputs::Vector<dim> &inputs,
  std::vector<dealii::Vector<double>> &               computed_quantities) const
{
  const unsigned int n_quadrature_points = inputs.solution_values.size();

  // Here inputs.solution_values correspond to the displacement,
  // more specifically, the locally relevant displacement field.
  Assert(inputs.solution_values[0].size() == dim * atomicity,
         dealii::ExcInternalError());
  Assert(computed_quantities.size() == n_quadrature_points,
         dealii::ExcInternalError());

  for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
      computed_quantities[q] = 0.; // initialize Values

      for (unsigned int a = 0; a < atomicity; ++a)
        for (auto d = 0; d < dim; ++d)
          computed_quantities[q](d) +=
            inputs.solution_values[q](d + dim * a) * atomicity_charges[a];
    }
}

#endif // __dealii_qc_compute_polarization_h
