
#include <deal.II-qc/potentials/potential_field.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
PotentialField<spacedim>::PotentialField (const bool   is_electric_field,
                                          const double initial_time)
  :
  FunctionTime<double>(initial_time),
  is_electric_field (is_electric_field)
{}



template <int spacedim>
PotentialField<spacedim>::~PotentialField()
{}



template class PotentialField<1>;
template class PotentialField<2>;
template class PotentialField<3>;


DEAL_II_QC_NAMESPACE_CLOSE
