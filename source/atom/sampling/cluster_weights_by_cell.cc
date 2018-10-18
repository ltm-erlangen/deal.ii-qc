
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{


  template <int dim, int atomicity, int spacedim>
  WeightsByCell<dim, atomicity, spacedim>::
  WeightsByCell (const double &cluster_radius,
                 const double &maximum_cutoff_radius)
    :
    WeightsByBase<dim, atomicity, spacedim>(cluster_radius, maximum_cutoff_radius)
  {}



  template <int dim, int atomicity, int spacedim>
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
  WeightsByCell<dim, atomicity, spacedim>::
  update_cluster_weights
  (const Triangulation<dim, spacedim>                               &triangulation,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const
  {
    const unsigned int n_sampling_points =
      WeightsByBase<dim, atomicity, spacedim>::n_sampling_points();

    AssertThrow (n_sampling_points == triangulation.n_vertices(),
                 ExcNotImplemented());

    // Prepare energy molecules in this container.
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    cell_energy_molecules;

    // Prepare the total number of molecules per cell in this container.
    // The container should also contain the information of total number of
    // molecules per cell for ghost cells on the current MPI process.
    std::map<types::CellIteratorType<dim, spacedim>, unsigned int>
    n_molecules_per_cell;

    // Prepare the number of cluster molecules per cell in this container.
    std::map<types::CellIteratorType<dim, spacedim>, unsigned int>
    n_cluster_molecules_per_cell;

    // Get the squared_energy_radius to identify energy molecules.
    const double squared_energy_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::maximum_cutoff_radius +
       WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Get the squared_cluster_radius to identify cluster molecules.
    const double squared_cluster_radius =
      dealii::Utilities::fixed_power<2>
      (WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Loop over all active cells of the mesh and initialize
    // n_molecules_per_cell and n_cluster_molecules_per_cell.
    for (types::CellIteratorType<dim, spacedim>
         cell  = triangulation.begin_active();
         cell != triangulation.end();
         cell++)
      {
        n_molecules_per_cell[cell]         = 0;
        n_cluster_molecules_per_cell[cell] = 0;
      }

    types::CellIteratorType<dim, spacedim> unique_cell =
      cell_molecules.begin()->first;

    // Prepare sampling points of this cell in this container.
    std::vector<Point<spacedim> > this_cell_sampling_points =
      WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(unique_cell);

    // Loop over all molecules, see if a given molecule is energy molecule and
    // if so if it's a cluster molecule.
    // While there, count the total number of molecules per cell and
    // number of cluster molecules per cell.
    for (const auto &cell_molecule : cell_molecules)
      {
        const auto &cell         = cell_molecule.first;
        Molecule<spacedim, atomicity> molecule = cell_molecule.second;

        if (unique_cell != cell)
          {
            unique_cell = cell;

            this_cell_sampling_points =
              WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(unique_cell);
          }

        Assert (n_molecules_per_cell.find(cell) !=n_molecules_per_cell.end(),
                ExcMessage("Provided 'mesh' isn't consistent with "
                           "the cell based molecules data structure."));

        n_molecules_per_cell[cell]++;

        // Get the global index of the sampling point (of this cell) closest
        // to the molecule and the squared distance of separation.
        const double squared_distance_from_closest_sampling_point =
          Utilities::
          find_closest_point (molecule_initial_location(molecule),
                              this_cell_sampling_points).second;

        if (squared_distance_from_closest_sampling_point < squared_energy_radius)
          {
            if (squared_distance_from_closest_sampling_point < squared_cluster_radius)
              {
                // Increment cluster molecule count for this "cell"
                n_cluster_molecules_per_cell[cell]++;
                // molecule is cluster molecule
                molecule.cluster_weight = 1.;
              }
            else
              // molecule is not cluster molecule
              molecule.cluster_weight = 0.;

            // Insert molecule into cell_energy_molecules as the distance
            // from the molecule to one of the sampling points of the cell is
            // less than energy_radius.
            cell_energy_molecules.insert(std::make_pair(cell, molecule));
          }
      }

    //---Finished adding energy molecules
    //---Now update cluster weights with correct value

    // Loop over all the energy molecules,
    // update their weights by multiplying with the factor
    // (n_molecules/n_cluster_molecules)
    for (auto &energy_molecule : cell_energy_molecules)
      {

        Assert (n_cluster_molecules_per_cell.at(energy_molecule.first) != 0,
                ExcInternalError());

        // The cluster weight was previously set to 1. if the molecule is
        // cluster molecule and 0. if the molecule is not cluster molecule.
        energy_molecule.second.cluster_weight *=
          static_cast<double>(n_molecules_per_cell.at(energy_molecule.first))
          /
          static_cast<double>(n_cluster_molecules_per_cell.at(energy_molecule.first));
      }

    //---Check in Debug mode that n_molecules and n_energy_molecules computed here
    //   are indeed similar to what CellMoleculeTools functions return.
    //   The reason for not using CellMoleculeTools functions is that
    //   the code here is already optimized and tested. So we assert that
    //   CellMoleculeTools functions also yield same result.
#ifdef Debug
    for (types::CellIteratorType<dim, spacedim>
         cell  = triangulation.begin_active();
         cell != triangulation.end();
         cell++)
      {
        // Get n_molecules_in_cell using molecules_range.second
        const auto molecules_range =
          CellMoleculeTools::molecules_range_in_cell(cell, cell_molecules);

        // Get the number of cluster molecules in cell.
        // It is legal to call this function as we have
        // already updated cluster weights
        const auto n_cluster_molecules_in_cell =
          CellMoleculeTools::n_cluster_molecules_in_cell(cell, cell_energy_molecules);

        Assert (n_molecules_per_cell[cell] == molecules_range.second,
                ExcInternalError());

        Assert (n_cluster_molecules_per_cell[cell] == n_cluster_molecules_in_cell,
                ExcInternalError());
      }
#endif

    return cell_energy_molecules;
  }



#define SINGLE_WEIGHTS_BY_CELL_INSTANTIATION(_DIM_, _ATOMICITY_, _SPACE_DIM_) \
  template class WeightsByCell< _DIM_, _ATOMICITY_, _SPACE_DIM_ >;            \
   
#define WEIGHTS_BY_CELL(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,         \
              SINGLE_WEIGHTS_BY_CELL_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X              \
   
  // WeightsByCell class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_CELL)

#undef SINGLE_WEIGHTS_BY_CELL_INSTANTIATION
#undef WEIGHTS_BY_CELL

} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
