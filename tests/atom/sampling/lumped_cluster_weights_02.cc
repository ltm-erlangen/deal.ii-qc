#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <deal.II-qc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



// Compute the mass of the atomistic system of NaCl nano-crystal of 4096 atoms
// using lumped clustering approach.



class TuneConfigureQC : public ConfigureQC
{
public:

  TuneConfigureQC(std::shared_ptr<std::istream> is)
    :
    ConfigureQC(is)
  {}

  // Change the cluster radius
  void set_cluster_radius(const double &cluster_radius)
  {
    ConfigureQC::cluster_radius = cluster_radius;
  }

};



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const ConfigureQC &config)
    :
    QC<dim, PotentialType>(config)
  {}

  void partial_run (const ConfigureQC &config);
};



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run(const ConfigureQC &config)
{
  QC<dim, PotentialType>::configure_qc = config;
  QC<dim, PotentialType>::setup_energy_atoms_with_cluster_weights();

  const auto &energy_atoms =  QC<dim, PotentialType>::atom_data.energy_atoms;

  const double cluster_radius =
    QC<dim, PotentialType>::configure_qc.get_cluster_radius();

  double mass = 0.;

  for (const auto &cell_atom : energy_atoms)
    if (cell_atom.first->is_locally_owned())
      mass += cell_atom.second.cluster_weight;

  dealii::Utilities::MPI::sum (mass,
                               QC<dim, PotentialType>::mpi_communicator);

  QC<dim, PotentialType>::pcout << cluster_radius << "\t"
                                << mass << "\t"
                                << std::endl;

}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                                  dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss << "set Dimension = 3"                              << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = 7.5"                        << std::endl
          << "    set Y center = 7.5"                        << std::endl
          << "    set Z center = 7.5"                        << std::endl
          << "    set X extent = 15."                         << std::endl
          << "    set Y extent = 15."                         << std::endl
          << "    set Z extent = 15."                         << std::endl
          << "    set X repetitions = 1"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 2" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                   << std::endl
          << "  set Maximum cutoff radius = 5"              << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../../data/16_NaCl_atom.data"     << std::endl
          << "end"                                          << std::endl
          << "subsection Configure QC"                      << std::endl
          << "  set Ghost cell layer thickness = 5.01"      << std::endl
          << "  set Cluster radius = 1"                     << std::endl
          << "  set Cluster weights by type = LumpedVertex" << std::endl
          << "end"                                          << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      TuneConfigureQC tune_config( prm_stream );
      // Define Problem
      Problem<3, Potential::PairCoulWolfManager> problem(tune_config);

      double new_cluster_radius = 1.0;

      for (unsigned int i = 0; i < 16; ++i, new_cluster_radius += 1)
        {
          tune_config.set_cluster_radius(new_cluster_radius);
          problem.partial_run (tune_config);
        }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}
