#ifndef __dealii_qc_atom_handler_h
#define __dealii_qc_atom_handler_h


#include <dealiiqc/atom/atom_handler.h>
#include <dealiiqc/qc.h>

namespace dealiiqc
{
  using namespace dealii;

  template <int dim>
  struct CellData
  {

    /**
     * A typedef for active_cell_iterator for ease of use
     */
    typedef typename parallel::shared::Triangulation<dim>::active_cell_iterator CellIterator;

    /**
     * FEValues object to evaluate fields and shape function values at
     * quadrature points.
     */
    std::shared_ptr<FEValues<dim>> fe_values;

    /**
     * Any atom contributing to (QC) energy of the system is in @see energy_atoms.
     * An atom contributes to (QC) energy computations if it happens to be
     * located within a distance of @see cluster_radius + @see cutoff_radius
     * to any vertex.
     */
    std::multimap< CellIterator, Atom<dim>> energy_atoms;

    /**
     * A map of global atom IDs to quadrature point (local id of at atom)
     */
    std::map<unsigned int, unsigned int> quadrature_atoms;

    /**
     * A vector to store displacements evaluated at quadrature points
     */
    mutable std::vector<Tensor<1,dim>> displacements;

    /**
     * A map for each cell to related global degree-of-freedom, to those
     * defined on the cell. Essentially, the reverse of
     * cell->get_dof_indices().
     */
    std::map<unsigned int, unsigned int> global_to_local_dof;

  };

  namespace Cluster
  {

    /**
     * Base class for assigning @see cluster_weight to atoms
     */
    template<int dim>
    class WeightsByBase
    {
    public:

      virtual ~WeightsByBase();

      /**
       * A typedef for active_cell_iterator for ease of use
       */
      using CellIterator = typename parallel::shared::Triangulation<dim>::active_cell_iterator;

      /**
       * A typedef for cell and atom associations
       */
      using CellAtomIterator = typename std::multimap< CellIterator, Atom<dim>>::iterator;

      /**
       * A typedef for cluster atoms
       */
      using ClusterAtomIterator = typename std::multimap< CellIterator, CellAtomIterator>;

      /**
       * Function through which cluster_weights are assigned to atoms.
       */
      virtual void update_atom_cluster_weights();
    };

    template<int dim>
    class WeightsByCell : public WeightsByBase<dim>
    {
    public:

      /**
       * Using typedefs used in WeightsByBase
       */
      using CellIterator = typename WeightsByBase<dim>::CellIterator;
      using CellAtomIterator = typename WeightsByBase<dim>::CellAtomIterator;
      using ClusterAtomIterator = typename WeightsByBase<dim>::ClusterAtomIterator;

      /**
       * Update cluster atom's @see cluster_weight per cell (one cell per function call).
       * @param[in] cell points to the cell of whose cluster atom's we are updating weights.
       * @param[in] cell_atom_count the total number of atom count inside this cell
       * (without discounting atoms with special attribute namely,
       * cluster_atom or energy_atom)
       * @param[out] cluster_atoms contains iterators to the cluster atoms
       * (whose weights) will be updated by this function.
       * Returns the number of cluster atoms in the cell.
       */
      types::global_atom_index
      update_weights_per_cell_( const CellIterator &cell,
                                const types::global_atom_index &n_cell_atoms,
                                ClusterAtomIterator &cluster_atoms)
      {
        types::global_atom_index cluster_atom_count_in_cell =
          static_cast<types::global_atom_index> (cluster_atoms.count(cell));

        auto atom_range = cluster_atoms.equal_range(cell);

        types::global_atom_index cluster_atom_counter =0;
        for ( auto it = atom_range.first; it != atom_range.second; ++it )
          {
            (*it).cluster_weight = static_cast<double>( n_cell_atoms)/
                                   static_cast<double>(cluster_atom_count_in_cell);
            cluster_atom_counter++;
          }

        Assert( cluster_atom_counter == cluster_atom_count_in_cell,
                ExcInternalError());

        return cluster_atom_count_in_cell;
      }
    };

  }


} /* namespace dealiiqc */

#endif // __dealii_qc_atom_handler_h
