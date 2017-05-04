
#ifndef __dealii_qc_pair_base_h
#define __dealii_qc_pair_base_h

#include <dealiiqc/potentials/potentials.h>
#include <dealiiqc/utilities.h>

namespace dealiiqc
{

  namespace Potential
  {

    /**
     * Base class of the pair potential classes.
     */
    class PairBaseManager
    {

    public:

      /**
       * Set the charges of each atom type in the system using a shared pointer
       * to a list of @p charges. This is needed by the
       * derived classes to compute Coulomb interaction energy when there are
       * atoms with non-zero charges in the system.
       *
       * In typical usage case we do not know all the charges until we parse
       * the atom data. However, we want to create an object of the derived
       * class before we start parsing the atom data. Therefore the
       * charges in system are set-up in a separate function as opposed to the
       * constructor of derived classes.
       */
      void set_charges (std::shared_ptr<std::vector<types::charge>> &charges_);

    protected:

      /**
       * A shared pointer to the list of charges in the system.
       */
      std::shared_ptr<std::vector<types::charge>> charges;


    };

  } // namespace Potential

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_lj_cut_h */
