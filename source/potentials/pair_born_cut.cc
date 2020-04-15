
#include <deal.II/base/utilities.h>

#include <deal.II-qc/potentials/pair_born_cut.h>

#include <limits>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{
  PairBornCutManager::PairBornCutManager(const double &cutoff_radius)
    : cutoff_radius_squared(cutoff_radius * cutoff_radius)
  {}



  void
  PairBornCutManager::declare_interactions(
    const types::atom_type     i_atom_type,
    const types::atom_type     j_atom_type,
    const InteractionTypes     interaction,
    const std::vector<double> &parameters)
  {
    Assert(interaction == InteractionTypes::Born,
           ExcMessage("Invalid InteractionTypes specified"));
    Assert(parameters.size() == 5, ExcMessage("Invalid parameters list"));

    const std::array<double, 5> params = {parameters[0],
                                          1.0 / parameters[1],
                                          parameters[2],
                                          parameters[3],
                                          parameters[4]};

    born_parameters.insert(
      std::make_pair(get_pair(i_atom_type, j_atom_type), params));

    DEAL_II_QC_UNUSED_VARIABLE(interaction);
  }


} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
