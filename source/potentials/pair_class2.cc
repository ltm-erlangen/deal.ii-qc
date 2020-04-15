
#include <deal.II/base/utilities.h>

#include <deal.II-qc/potentials/pair_class2.h>

#include <limits>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Potential
{
  PairClass2Manager::PairClass2Manager()
  {}



  void
  PairClass2Manager::declare_interactions(const types::atom_type i_atom_type,
                                          const types::atom_type j_atom_type,
                                          const InteractionTypes interaction,
                                          const std::vector<double> &parameters)
  {
    Assert(interaction == InteractionTypes::Class2,
           ExcMessage("Invalid InteractionTypes specified"));
    Assert(parameters.size() == 4, ExcMessage("Invalid parameters list"));

    const std::array<double, 4> params = {parameters[0],
                                          parameters[1],
                                          parameters[2],
                                          parameters[3]};

    class2_parameters.insert({get_pair(i_atom_type, j_atom_type), params});

    DEAL_II_QC_UNUSED_VARIABLE(interaction);
  }


} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
