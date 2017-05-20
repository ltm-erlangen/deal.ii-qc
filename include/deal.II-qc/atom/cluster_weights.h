
#ifndef __dealii_qc_cluster_weights_h_
#define __dealii_qc_cluster_weights_h_

#include <deal.II-qc/atom/atom_data.h>

namespace dealiiqc
{

  namespace Cluster
  {

    /**
     * Base class for assigning @see cluster_weight to atoms
     */
    template <int dim>
    class WeightsByBase
    {
    public:

      /**
       * Constructor.
       */
      WeightsByBase( const double &cluster_radius);


      virtual ~WeightsByBase();

      /**
       * Calculate and assign cluster weights to to all cluster atoms in
       * @p energy_atoms stored on this MPI process using, if needed,
       * additionally provided number of disregarded atoms in the fully
       * resolved simulation @p n_thrown_atoms_per_cell
       */
      virtual
      void
      update_cluster_weights (const std::map< types::CellIteratorType<dim>, unsigned int> &n_thrown_atoms_per_cell,
                              types::CellAtomContainerType<dim> &energy_atoms) const = 0;

    protected:

      /**
       * The cluster radius for the QC approach.
       */
      const double cluster_radius;
    };

    /**
     * A derived class for updating cluster weights using the cell approach.
     */
    template <int dim>
    class WeightsByCell : public WeightsByBase <dim>
    {
    public:

      /**
       * Constructor
       */
      WeightsByCell(const double &cluster_radius);

      void
      update_cluster_weights( const std::map< types::CellIteratorType<dim>, unsigned int> &n_thrown_atoms_per_cell,
                              types::CellAtomContainerType<dim> &energy_atoms) const;

    };


  } // namespace Cluster


} // namespace dealiiqc



#endif /* __dealii_qc_cluster_weights_h_ */
