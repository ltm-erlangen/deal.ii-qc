#ifndef __dealii_qc_pair_lj_cut_h
#define __dealii_qc_pair_lj_cut_h

#include <dealiiqc/potentials/pair.h>

namespace dealiiqc
{

  namespace Potential
  {
    /**
     * Truncated Lennard-Jones pair potential.
     */
    template<int dim>
    class PairLJCut : public Pair<dim>
    {

      public:

        /**
         * Constructor.
         */
        PairLJCut( const ConfigureQC &config );

        /**
         * Destructor.
         */
        ~PairLJCut(){}

        double value ( const types::NeighborListsType<dim> &neighbor_lists);

        void gradient ( vector_t &gradient,
                        const types::CellAssemblyData<dim> &cell_assembly_data,
                        const types::NeighborListsType<dim> &neighbor_lists) const;

        void parse_pair_coefficients ();
    };

  }

} /* namespace dealiiqc */

#endif /* __dealii_qc_pair_lj_cut_h */
