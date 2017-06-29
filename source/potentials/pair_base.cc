
#include <deal.II-qc/potentials/pair_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Potential
{


  PairBaseManager::PairBaseManager ()
  {
    charges = NULL;
  }



  PairBaseManager::~PairBaseManager () {}



  void
  PairBaseManager::set_charges (std::shared_ptr<std::vector<types::charge>> &charges_)
  {
    // non-virtual function
    // Update the shared pointer to point to the vector of charges
    // (whose size should be of size equal to the number of different
    //  atom types)
    charges = std::const_pointer_cast<const std::vector<types::charge>>(charges_);
  }



} // namespace Potential


DEAL_II_QC_NAMESPACE_CLOSE
