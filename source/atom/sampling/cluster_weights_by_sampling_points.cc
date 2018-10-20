
#include <deal.II/base/utilities.h>

#include <deal.II-qc/atom/sampling/cluster_weights_by_sampling_points.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{
  template <int dim, int atomicity, int spacedim>
  WeightsBySamplingPoints<dim, atomicity, spacedim>::WeightsBySamplingPoints(
    const double &cluster_radius,
    const double &maximum_cutoff_radius)
    : WeightsByBase<dim, atomicity, spacedim>(cluster_radius,
                                              maximum_cutoff_radius)
  {}



  template <int dim, int atomicity, int spacedim>
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
  WeightsBySamplingPoints<dim, atomicity, spacedim>::update_cluster_weights(
    const Triangulation<dim, spacedim> &triangulation,
    const types::CellMoleculeContainerType<dim, atomicity, spacedim>
      &cell_molecules) const
  {
    // Prepare energy molecules in this container.
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
      cell_energy_molecules;

    const unsigned int n_sampling_points =
      WeightsByBase<dim, atomicity, spacedim>::n_sampling_points();

    const parallel::Triangulation<dim, spacedim> *const ptria =
      dynamic_cast<const parallel::Triangulation<dim, spacedim> *>(
        &triangulation);

    // Get a consistent MPI_Comm.
    const MPI_Comm &mpi_communicator =
      ptria != nullptr ? ptria->get_communicator() : MPI_COMM_SELF;

    // Prepare the total number of molecules per sampling point in this
    // container. The container should also contain the information of
    // the total number of molecules per sampling point for ghost cells of
    // the current MPI process.
    std::vector<unsigned int> n_molecules_per_sampling_point(n_sampling_points,
                                                             0);

    // Prepare the number of cluster molecules per sampling point in this
    // container.
    std::vector<unsigned int> n_cluster_molecules_per_sampling_point(
      n_sampling_points, 0);

    // Get the squared_energy_radius to identify energy molecules.
    const double squared_energy_radius = dealii::Utilities::fixed_power<2>(
      WeightsByBase<dim, atomicity, spacedim>::maximum_cutoff_radius +
      WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    // Get the squared_cluster_radius to identify cluster molecules.
    const double squared_cluster_radius = dealii::Utilities::fixed_power<2>(
      WeightsByBase<dim, atomicity, spacedim>::cluster_radius);

    types::CellIteratorType<dim, spacedim> unique_cell =
      cell_molecules.begin()->first;

    // Get the global indices of the sampling points of this cell.
    std::vector<unsigned int> this_cell_sampling_indices =
      WeightsByBase<dim, atomicity, spacedim>::get_sampling_indices(
        unique_cell);

    // Prepare sampling points of this cell in this container.
    std::vector<Point<spacedim>> this_cell_sampling_points =
      WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(unique_cell);

    // Loop over all molecules, see if a given molecules is energy molecules and
    // if so if it's a cluster molecules.
    // While there, count the total number of molecules per sampling point and
    // number of cluster molecules per sampling point.
    for (const auto &cell_molecule : cell_molecules)
      {
        const auto &                  cell     = cell_molecule.first;
        Molecule<spacedim, atomicity> molecule = cell_molecule.second;

        if (unique_cell != cell)
          {
            unique_cell = cell;

            this_cell_sampling_indices =
              WeightsByBase<dim, atomicity, spacedim>::get_sampling_indices(
                unique_cell);

            this_cell_sampling_points =
              WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(
                unique_cell);
          }

        // Get the global index of the sampling point (of this cell) closest
        // to the molecule and the squared distance of separation.
        const std::pair<unsigned int, double> closest_sampling_point =
          Utilities::find_closest_point(molecule_initial_location(molecule),
                                        this_cell_sampling_points);

        const unsigned int closest_sampling_index =
          // Advance from begin to location_in_container to get the closest
          // sampling point index.
          *std::next(this_cell_sampling_indices.cbegin(),
                     closest_sampling_point.first);

        // Squared distance to the closest sampling point.
        const double &squared_distance = closest_sampling_point.second;

        // Count only molecules assigned to locally owned cells, so that
        // when sum is performed over all MPI processes only the molecules
        // assigned locally owned cells are counted.
        if (cell->is_locally_owned())
          n_molecules_per_sampling_point[closest_sampling_index]++;

        if (squared_distance < squared_energy_radius)
          {
            if (squared_distance < squared_cluster_radius)
              {
                // Count only the cluster molecules of the locally owned cells.
                if (cell->is_locally_owned())
                  // Increment cluster molecules count for this "vertex"
                  n_cluster_molecules_per_sampling_point
                    [closest_sampling_index]++;
                // molecules is cluster molecules
                molecule.cluster_weight = 1.;
              }
            else
              // molecules is not cluster molecules
              molecule.cluster_weight = 0.;

            // Insert molecules into cell_energy_molecules if it is within
            // a distance of energy radius to associated cell's vertices.
            cell_energy_molecules.insert(std::make_pair(cell, molecule));
          }
      }

    //---Finished adding energy molecules

    // Accumulate the number of molecules per vertex from all MPI processes.
    dealii::Utilities::MPI::sum(n_molecules_per_sampling_point,
                                mpi_communicator,
                                n_molecules_per_sampling_point);

    // Accumulate the number of cluster molecules per vertex from all MPI
    // processes.
    dealii::Utilities::MPI::sum(n_cluster_molecules_per_sampling_point,
                                mpi_communicator,
                                n_cluster_molecules_per_sampling_point);

    //---Now update cluster weights with correct value

    unique_cell = cell_energy_molecules.begin()->first;

    // Get the global indices of the sampling points of this cell.
    this_cell_sampling_indices =
      WeightsByBase<dim, atomicity, spacedim>::get_sampling_indices(
        unique_cell);

    // Prepare sampling points of this cell in this container.
    this_cell_sampling_points =
      WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(unique_cell);

    // Loop over all the energy molecules,
    // update their weights by multiplying with the factor
    // (n_molecules/n_cluster_molecules)
    for (auto &energy_molecule : cell_energy_molecules)
      {
        const auto &                   cell     = energy_molecule.first;
        Molecule<spacedim, atomicity> &molecule = energy_molecule.second;

        if (unique_cell != cell)
          {
            unique_cell = cell;

            this_cell_sampling_indices =
              WeightsByBase<dim, atomicity, spacedim>::get_sampling_indices(
                unique_cell);

            this_cell_sampling_points =
              WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(
                unique_cell);
          }

        // Get the closest sampling point of the cell to the given point.
        const unsigned int location_in_container =
          Utilities::find_closest_point(molecule_initial_location(molecule),
                                        this_cell_sampling_points)
            .first;

        const unsigned int closest_sampling_index =
          // Advance from begin to location_in_container to get the closest
          // sampling point index.
          *std::next(this_cell_sampling_indices.cbegin(),
                     location_in_container);

        Assert(n_cluster_molecules_per_sampling_point[closest_sampling_index] !=
                 0,
               ExcInternalError());

        // The cluster weight was previously set to 1. if the molecules is
        // cluster molecules and 0. if the molecules is not cluster molecules.
        molecule.cluster_weight *=
          static_cast<double>(
            n_molecules_per_sampling_point[closest_sampling_index]) /
          static_cast<double>(
            n_cluster_molecules_per_sampling_point[closest_sampling_index]);
      }

    return cell_energy_molecules;
  }



#define SINGLE_WEIGHTS_BY_SPOINTS_INSTANTIATION(_DIM, _ATOMICITY, _SPACE_DIM) \
  template class WeightsBySamplingPoints<_DIM, _ATOMICITY, _SPACE_DIM>;

#define WEIGHTS_BY_SAMPLING_POINTS(R, X)               \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,            \
              SINGLE_WEIGHTS_BY_SPOINTS_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3))                   \
  X

  // WeightsBySamplingPoints class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_SAMPLING_POINTS)

#undef SINGLE_WEIGHTS_BY_VERTEX_INSTANTIATION
#undef WEIGHTS_BY_VERTEX


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
