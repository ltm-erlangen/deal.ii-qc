
#include <deal.II/base/utilities.h>

#include <deal.II-qc/potentials/pair_lj_cut.h>

#include <limits>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{
  PairLJCutManager::PairLJCutManager(const double &cutoff_radius,
                                     const bool    with_tail)
    : cutoff_radius_squared(cutoff_radius * cutoff_radius)
    , with_tail(with_tail)
  {}

  void
  PairLJCutManager::declare_interactions(const types::atom_type     i_atom_type,
                                         const types::atom_type     j_atom_type,
                                         const InteractionTypes     interaction,
                                         const std::vector<double> &parameters)
  {
    Assert(interaction == InteractionTypes::LJ,
           ExcMessage("Invalid InteractionTypes specified"));
    Assert(parameters.size() == 2, ExcMessage("Invalid parameters list"));
    Assert(parameters[0] >= 0.,
           ExcMessage("Invalid epsilon value specified for LJ pair potential"));
    Assert(parameters[1] > 0.,
           ExcMessage("Invalid r_m value specified for LJ pair potential"));

    const std::array<double, 2> params = {
      {parameters[0], dealii::Utilities::fixed_power<6>(parameters[1])}};

    // Replace if alraedy existing, insert if not existing.
    lj_parameters[get_pair(i_atom_type, j_atom_type)] = params;

    DEAL_II_QC_UNUSED_VARIABLE(interaction);
  }


} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
