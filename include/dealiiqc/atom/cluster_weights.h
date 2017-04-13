
#ifndef __dealii_qc_cluster_weights_h_
#define __dealii_qc_cluster_weights_h_

#include <dealiiqc/atom/atom_handler.h>

namespace dealiiqc
{

  namespace Cluster
  {

    /**
     * Base class for assigning @see cluster_weight to atoms
     */
    template<int dim>
    class WeightsByBase
    {
    public:

      /**
       * Constructor
       */
      WeightsByBase( const ConfigureQC &config);


      virtual ~WeightsByBase() {}

      /**
       * Function through which cluster_weights are assigned to atoms.
       */
      virtual void update_cluster_weights( const std::map< typename AtomHandler<dim>::CellIteratorType, unsigned int> &n_thrown_atoms_per_cell,
                                           typename AtomHandler<dim>::CellAtomContainerType &energy_atoms)=0;

    protected:
      const ConfigureQC &config;
    };

    template<int dim>
    class WeightsByCell : public WeightsByBase<dim>
    {
    public:

      /**
       * Constructor
       */
      WeightsByCell(const ConfigureQC &config);

      /**
       * Update cluster weights of the cluster atoms in @p energy_atoms
       * using @p n_thrown_atoms_per_cell.
       */
      void
      update_cluster_weights( const std::map< typename AtomHandler<dim>::CellIteratorType, unsigned int> &n_thrown_atoms_per_cell,
                              typename AtomHandler<dim>::CellAtomContainerType &energy_atoms);

    };

  }


}



#endif /* __dealii_qc_cluster_weights_h_ */
