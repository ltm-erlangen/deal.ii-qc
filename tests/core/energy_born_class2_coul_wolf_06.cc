
#include <deal.II/numerics/data_out.h>

#include <deal.II-qc/atom/compute_polarization.h>

#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of eight BaTiO3 molecules,
// whose atoms interact through BornCutClass2CoulWolfManager,
// using QC approach with full atomistic resolution.
// Because the distance between cores and shells of respective atoms is zero,
// the bond energy is zero.
// The blessed output is created using LAMMPS python script.
// The script is included at the end.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run();
  void
  displace_shells(const double shift);
  void
  compute_polarization();
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::displace_shells(const double shift)
{
  auto &displacement = this->locally_relevant_displacement;

  const auto         solution_size = displacement.size();
  const unsigned int dofs_per_cell = this->fe.dofs_per_cell;
  Assert(dofs_per_cell == solution_size, ExcInternalError());

  const auto division = std::div(dofs_per_cell, dim * atomicity);
  Assert(division.rem == 0, ExcInternalError());

  const unsigned int n_quadrature_points = division.quot;

  bool skip  = true;
  auto entry = displacement.begin();
  while (entry != displacement.end())
    {
      for (auto d = 0;
           d < n_quadrature_points * dim && entry != displacement.end();
           ++d, ++entry)
        if (skip == false) // Do not skip shells
          *entry = shift;

      skip = !skip;
    };

  displacement.compress(VectorOperation::insert);
}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::compute_polarization()
{
  DataOut<dim> data_out;

  // Prepare list of charges based on atomicity
  std::array<dealiiqc::types::charge, atomicity> atomicity_charges;

  const auto &mol_charges = *this->cell_molecule_data.charges;
  const auto &atoms =
    this->cell_molecule_data.cell_energy_molecules.begin()->second.atoms;

  for (auto a = 0; a < atomicity; ++a)
    atomicity_charges[a] = mol_charges[atoms[a].type];

  ComputePolarization<dim, atomicity> postprocessor(atomicity_charges);
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(this->locally_relevant_displacement, postprocessor);
  data_out.build_patches();

  std::cout << "\n\n#--- Polarization values in GNUPLOT format:" << std::endl;
  std::cout << "#" << std::endl;
  data_out.write_gnuplot(std::cout);
}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run()
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->update_neighbor_lists();

  MPI_Barrier(this->mpi_communicator);

  Testing::SequentialFileStream write_sequentially(this->mpi_communicator);

  const auto &cell_molecules = this->cell_molecule_data.cell_energy_molecules;

  const double energy =
    this->template compute<false>(this->locally_relevant_gradient);

  this->pcout << "The energy computed using PairBornCutClass2CoulWolfManager "
              << "of atomistic system is: " << std::fixed
              << std::setprecision(3) << energy << " eV." << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(this->neighbor_lists.size(),
                                this->mpi_communicator);

  this->pcout << "Total number of neighbors " << total_n_neighbors << std::endl;

  for (auto entry : this->neighbor_lists)
    this->pcout << "Molecule I: " << entry.second.first->second.global_index
                << '\t'
                << "Molecule J: " << entry.second.second->second.global_index
                << std::endl;
  this->displace_shells(1.);
  this->compute_polarization();
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 2." << std::endl
          << "    set Y center = 2." << std::endl
          << "    set Z center = 2." << std::endl
          << "    set X extent = 4." << std::endl
          << "    set Y extent = 4." << std::endl
          << "    set Z extent = 4." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Number of atom types = 6" << std::endl
          << "  set Maximum cutoff radius = 100" << std::endl
          << "  set Pair potential type = Born Class2 Coul Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 14.5, 16.0" << std::endl
          << "  set Factor coul = 0." << std::endl
          << "  set Pair specific coefficients = "
             "*, *,    0.00,	1.0000,	0.000,	0.0000,	0.000;"
             "2, 6, 7149.81,	0.3019,	0.000,	0.0000,	0.000;"
             "4, 6, 7200.27,	0.2303,	0.000,	0.0000,	0.000;"
             "6, 6, 3719.60,	0.3408,	0.000,	597.17,	0.000;"
          << std::endl
          << "  set Bond type = Class2" << std::endl
          << "  set Bond specific coefficients = "
             "1,  2,  0.000, 149.255, 0.0,   0.0000000;"
             "3,  4,  0.000, 153.070, 0.0,  20.83333333;"
             "5,  6,  0.000,  18.465, 0.0, 208.33333333;"
          << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/BaTiO3_cs_2x1x1_qcatom.data" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = -1." << std::endl
          << "  set Cluster radius = 100" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairBornCutClass2CoulWolfManager, 10> problem(
        config);
      problem.partial_run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
