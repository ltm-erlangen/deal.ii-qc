
#include <dealiiqc/potentials/pair_base.h>



namespace dealiiqc
{

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
      charges = charges_;
    }



  } // namespace Potential

} // namespace dealiiqc
