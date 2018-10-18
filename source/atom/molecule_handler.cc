
#include <deal.II/distributed/shared_tria.h>

#include <deal.II-qc/atom/molecule_handler.h>


DEAL_II_QC_NAMESPACE_OPEN


template<int dim, int atomicity, int spacedim>
MoleculeHandler<dim, atomicity, spacedim>::
MoleculeHandler (const ConfigureQC &configure_qc)
  :
  configure_qc(configure_qc)
{}



template<int dim, int atomicity, int spacedim>
types::CellMoleculeNeighborLists<dim, atomicity, spacedim>
MoleculeHandler<dim, atomicity, spacedim>::
get_neighbor_lists
(const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_energy_molecules) const
{
  // Check whether cell_energy_molecules is empty; throw error if it is.
  Assert (cell_energy_molecules.size(),
          ExcInternalError());

  types::CellMoleculeNeighborLists<dim, atomicity, spacedim> neighbor_lists;

  // cell_neighbor_lists contains all the pairs of cell
  // whose molecules interact with each other.
  std::list< std::pair<
  types::CellIteratorType<dim, spacedim>
  ,
  types::CellIteratorType<dim, spacedim> >> cell_neighbor_lists;

  const double cutoff_radius  = configure_qc.get_maximum_cutoff_radius();
  const double squared_cutoff_radius  =
    dealii::Utilities::fixed_power<2>(cutoff_radius);

  // For each locally owned cell, identify all the cells
  // whose associated molecules may interact. At this point we do not
  // check if there are indeed some interacting molecules,
  // i.e. those within the cut-off radius. This is done to speedup
  // building of the neighbor list.
  // TODO: this approach strictly holds in the reference
  // (undeformed) configuration only.
  // It may still be OK for small deformations,
  // but for large deformations we may need to
  // use something like MappingQEulerian to work
  // with the deformed mesh.
  // TODO: optimize loop over unique keys
  // ( mulitmap::upper_bound()'s complexity is O(nlogn) )
  for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
       unique_I  = cell_energy_molecules.cbegin();
       unique_I != cell_energy_molecules.cend();
       unique_I  = cell_energy_molecules.upper_bound(unique_I->first))
    // Only locally owned cells have cell neighbors
    if (unique_I->first->is_locally_owned())
      {
        types::ConstCellIteratorType<dim, spacedim> cell_I = unique_I->first;

        // Get center and the radius of the enclosing ball of cell_I
        const auto enclosing_ball_I = cell_I->enclosing_ball();

        for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
             unique_J  = cell_energy_molecules.cbegin();
             unique_J != cell_energy_molecules.cend();
             unique_J  = cell_energy_molecules.upper_bound(unique_J->first))
          {
            types::ConstCellIteratorType<dim, spacedim>
            cell_J = unique_J->first;

            // Get center and the radius of the enclosing ball of cell_I
            const auto enclosing_ball_J = cell_J->enclosing_ball();

            // If the shortest distance between the enclosing balls of
            // cell_I and cell_J is less than cutoff_radius, then the
            // cell pair is in the cell_neighbor_lists.
            if (enclosing_ball_I.first.distance_square(enclosing_ball_J.first)
                < dealii::Utilities::fixed_power<2>(cutoff_radius +
                                                    enclosing_ball_I.second +
                                                    enclosing_ball_J.second) )
              cell_neighbor_lists.push_back( std::make_pair(cell_I, cell_J) );
          }
      }

  for (const auto &cell_pair_IJ : cell_neighbor_lists)
    {
      types::ConstCellIteratorType<dim, spacedim> cell_I = cell_pair_IJ.first;
      types::ConstCellIteratorType<dim, spacedim> cell_J = cell_pair_IJ.second;

      std::pair<
      types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      ,
      types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
      >
      range_of_cell_I = cell_energy_molecules.equal_range(cell_I),
      range_of_cell_J = cell_energy_molecules.equal_range(cell_J);

      // For each molecule associated to locally owned cell_I
      for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
           cell_molecule_I  = range_of_cell_I.first;
           cell_molecule_I != range_of_cell_I.second;
           cell_molecule_I++)
        {

          const Molecule<spacedim, atomicity>
          &molecule_I = cell_molecule_I->second;

          // Check if the molecule_I is cluster molecule,
          // only cluster molecules get to be in neighbor lists.
          if (molecule_I.cluster_weight != 0)

            for (types::CellMoleculeConstIteratorType<dim, atomicity, spacedim>
                 cell_molecule_J  = range_of_cell_J.first;
                 cell_molecule_J != range_of_cell_J.second;
                 cell_molecule_J++ )
              {

                const Molecule<spacedim, atomicity>
                &molecule_J = cell_molecule_J->second;

                // Check whether molecule_J is cluster molecule
                const bool
                molecule_J_is_cluster_molecule =
                  (molecule_J.cluster_weight != 0);

                // If molecule_J is also a cluster molecule,
                // then molecule_J could be only added to molecule_I's neighbor
                // list when molecule_I's index is greater than that of
                // molecule_J. This ensures that there is no double counting of
                // energy contribution thorugh both the cluster molecules:
                // molecule_I and molecule_J.
                // OR
                // If molecule_J is not cluster molecule,
                // then molecule_J could be added to (cluster) molecule_I's
                // neighbor list.
                if ( (molecule_J_is_cluster_molecule &&
                      (molecule_I.global_index > molecule_J.global_index))
                     ||
                     !molecule_J_is_cluster_molecule )

                  // Check distances between all atoms of one molecule to
                  // all atoms of the other molecule. If any two atoms of
                  // different molecules interact, then the molecules are in
                  // neighbor lists.
                  if (least_distance_squared(molecule_I, molecule_J)
                      < squared_cutoff_radius)
                    neighbor_lists.insert
                    (
                      std::make_pair (cell_pair_IJ,
                                      std::make_pair (cell_molecule_I,
                                                      cell_molecule_J))
                    );
              }
        }

    }

#ifdef DEBUG
  // For large deformations the cells might
  // distort leading to incorrect neighbor_list
  // using the above hierarchical procedure of
  // updating neighbor lists.
  // In debug mode, check if the number
  // of interactions is the same as computed
  // by the following.
  unsigned int total_number_of_interactions = 0;

  // loop over all atoms
  //   if locally owned cell
  //     loop over all atoms
  //       if within distance
  //         total_number_of_interactions++;
  for (const auto &cell_molecule_I : cell_energy_molecules)
    if (cell_molecule_I.first->is_locally_owned())
      {
        const Molecule<spacedim, atomicity>
        &molecule_I = cell_molecule_I.second;

        // Check if the molecule_I is cluster molecule,
        // only cluster molecules get to be in neighbor lists.
        if (molecule_I.cluster_weight != 0)
          for (const auto &cell_molecule_J : cell_energy_molecules)
            {
              const Molecule<spacedim, atomicity>
              &molecule_J = cell_molecule_J.second;

              // Check whether molecule_J is cluster molecule
              const bool
              molecule_J_is_cluster_molecule =
                (molecule_J.cluster_weight != 0);

              if ( (molecule_J_is_cluster_molecule &&
                    (molecule_I.global_index > molecule_J.global_index))
                   ||
                   !molecule_J_is_cluster_molecule )
                if (least_distance_squared(molecule_I, molecule_J)
                    < squared_cutoff_radius)
                  total_number_of_interactions++;
            }
      }
  Assert (total_number_of_interactions == neighbor_lists.size(),
          ExcMessage("Some of the interactions are not accounted "
                     "while updating neighbor lists"));
#endif

  return neighbor_lists;
}



#define SINGLE_MOLECULE_HANDLER_INSTANTIATION(_DIM_, _ATOMICITY_, _SPACE_DIM_) \
  template class MoleculeHandler< _DIM_, _ATOMICITY_, _SPACE_DIM_ >;           \
   
#define MOLECULE_HANDLER(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,          \
              SINGLE_MOLECULE_HANDLER_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X               \
   
// MoleculeHandler class Instantiations.
INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(MOLECULE_HANDLER)

#undef SINGLE_MOLECULE_HANDLER_INSTANTIATION
#undef MOLECULE_HANDLER


DEAL_II_QC_NAMESPACE_CLOSE
