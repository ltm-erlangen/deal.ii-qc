
#include <limits>

#include <deal.II/base/utilities.h>

#include <deal.II-qc/potentials/pair_lj_cut.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{


  PairLJCutManager::PairLJCutManager (const double &cutoff_radius)
    :
    cutoff_radius_squared(cutoff_radius *cutoff_radius)
  {}

  void
  PairLJCutManager::declare_interactions (const types::atom_type i_atom_type,
                                          const types::atom_type j_atom_type,
                                          const InteractionTypes interaction,
                                          const std::vector<double> &parameters)
  {
    Assert (interaction==InteractionTypes::LJ,
            ExcMessage("Invalid InteractionTypes specified"));
    Assert (parameters.size() == 2,
            ExcMessage("Invalid parameters list"));
    Assert (parameters[0] > 0.,
            ExcMessage("Invalid epsilon value specified for LJ pair potential"));
    Assert (parameters[1] > 0.,
            ExcMessage("Invalid r_m value specified for LJ pair potential"));

    const std::array<double, 2> params = {{parameters[0],dealii::Utilities::fixed_power<6>(parameters[1])}};
    lj_parameters.insert (std::make_pair (get_pair (i_atom_type,
                                                    j_atom_type),
                                          params) );
  }


} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
