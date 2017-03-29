
#include <dealiiqc/io/configure_qc.h>

namespace dealiiqc
{
  using namespace dealii;

  // Initialize dimension to a default unusable value
  // Imposes user to `set Dimension`
  ConfigureQC::ConfigureQC( std::shared_ptr<std::istream> is)
    :
    dimension(0),
    mesh_file(std::string()),
    n_initial_global_refinements(1),
    input_stream(is)
  {
    AssertThrow( *input_stream, ExcIO() );
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input (*input_stream,"dummy","#end-of-parameter-section");
    parse_parameters(prm);
  }

  unsigned int ConfigureQC::get_dimension() const
  {
    return dimension;
  }

  std::string ConfigureQC::get_mesh_file () const
  {
    return mesh_file;
  }

  std::string ConfigureQC::get_atom_data_file () const
  {
    return atom_data_file;
  }

  unsigned int ConfigureQC::get_n_initial_global_refinements() const
  {
    return n_initial_global_refinements;
  }

  std::shared_ptr<std::istream> ConfigureQC::get_stream() const
  {
    return input_stream;
  }

  std::string ConfigureQC::get_pair_style() const
  {
    return pair_style;
  }

  double ConfigureQC::get_cluster_radius() const
  {
    return cluster_radius;
  }

  double ConfigureQC::get_max_search_radius() const
  {
    return max_search_radius;
  }

  void ConfigureQC::declare_parameters( ParameterHandler &prm )
  {
    // TODO: Write intput file name to the screen
    //deallog << std::endl << "Parsing qc input file " << filename << std::endl
    //        << "for a " << dim << " dimensional simulation. " << std::endl;

    prm.declare_entry("Dimension", "2",
                      Patterns::Integer(1,3),
                      "Dimensionality of the problem ");
    prm.enter_subsection ("Configure mesh");
    {
      prm.declare_entry("Mesh file", "",
                        Patterns::Anything(),
                        "Name of the mesh file "
                        "with dealii compatible mesh files");
      prm.declare_entry("Number of initial global refinements", "1",
                        Patterns::Integer(0),
                        "Number of global mesh refinement cycles "
                        "applied to initial grid");
    }
    prm.leave_subsection ();

    // TODO: Declare atom information
    // Use LAMMPS-like atom data file
    prm.enter_subsection ("Configure atoms");
    {
      prm.declare_entry("Atom data file", "",
                        Patterns::Anything(),
                        "Name of the atom data file "
                        "that is compatible with LAMMPS");
      prm.declare_entry("Pair style", "coul/wolf 0.8 4.0",
                        Patterns::Anything(),
                        "Pairwise interactions style "
                        "potential energy function");
      prm.declare_entry("Pair coeff", "* *",
                        Patterns::Anything(),
                        "Coefficients for pair potential "
                        "and interaction between atom types");
    }
    prm.leave_subsection ();
    prm.enter_subsection ("Configure qc");
    {
      prm.declare_entry("Cluster radius", "2.0",
                        Patterns::Double(0),
                        "Cluster radius of each cluster "
                        "used to identify cluster atoms "
                        "for energy and force computations");
      prm.declare_entry("Max search radius", "6.0",
                        Patterns::Double(0),
                        "Maximum of all the cutoff radii "
                        "used to identify the ghost cells "
                        "of each MPI process");
    }
    prm.leave_subsection ();

    // TODO: Declare Run 0
    //       Compute energy and force at the initial configuration.

  }

  void ConfigureQC::parse_parameters( ParameterHandler &prm )
  {
    dimension = prm.get_integer("Dimension");
    prm.enter_subsection("Configure mesh");
    {
      mesh_file                    = prm.get("Mesh file");
      n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
    }
    prm.leave_subsection();
    prm.enter_subsection("Configure atoms");
    {
      atom_data_file = prm.get("Atom data file");
      pair_style = prm.get("Pair style");
    }
    prm.leave_subsection();
    prm.enter_subsection("Configure qc");
    {
      cluster_radius    = prm.get_double("Cluster radius");
      max_search_radius = prm.get_double("Max energy radius");
    }
    prm.leave_subsection();
  }

}
