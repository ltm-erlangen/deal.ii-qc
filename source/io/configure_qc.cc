
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

  double ConfigureQC::get_maximum_search_radius() const
  {
    return maximum_search_radius;
  }

  double ConfigureQC::get_maximum_energy_radius() const
  {
    return maximum_energy_radius;
  }

  double ConfigureQC::get_cluster_radius() const
  {
    return cluster_radius;
  }

  std::shared_ptr<Potential::PairBaseManager>
  ConfigureQC::get_potential() const
  {
    return pair_potential;
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
      prm.declare_entry("Maximum energy radius", "6.0",
                        Patterns::Double(0),
                        "Maximum of all the cutoff radii "
                        "plus a skin thickness "
                        "used to update the neighbor lists "
                        "of atoms");
      prm.declare_entry("Pair potential type", "LJ",
                        Patterns::Selection("Coulomb Wolf|LJ"),
                        "Pairwise interactions type of the "
                        "pair potential energy function");
      prm.declare_entry("Pair global coefficients", ".90",
                        Patterns::List(Patterns::Anything(),1),
                        "Comma separated global coefficient values for the "
                        "provided pair potential type."
                        "Coulomb Wolf: alpha and cutoff radius."
                        "LJ: cutoff radius ");
      prm.declare_entry("Pair specific coefficients", "0, 0, .8, 1.1;",
                        Patterns::List(Patterns::List(Patterns::Anything(),
                                                      0,
                                                      std::numeric_limits<unsigned int>::max(),
                                                      ","),
                                       0,
                                       std::numeric_limits<unsigned int>::max(),
                                       ";"),
                        "Additional coefficients for a pair of atoms of "
                        "certain types. Depending on the specific pair "
                        "potential type this input may not be necessary. "
                        "For the pair potential type: Coulomb Wolf, the pair "
                        "specific coefficients are not necessary."
                        "For the pair potential type: LJ, the first two "
                        "entries are the atom types (therefore of type "
                        "unsigned int) and the remaining two are epsilon and "
                        "alpha LJ parameters, respectively.");
    }
    prm.leave_subsection ();
    prm.enter_subsection ("Configure QC");
    {
      //TODO: Max->Maximum
      prm.declare_entry("Max search radius", "6.0",
                        Patterns::Double(0),
                        "Maximum of all the cutoff radii "
                        "used to identify the ghost cells "
                        "of each MPI process");
      prm.declare_entry("Cluster radius", "2.0",
                        Patterns::Double(0),
                        "Cluster radius used in "
                        "QC simulation");
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
      maximum_energy_radius = prm.get_double("Maximum energy radius");

      const std::string pair_potential_type = prm.get("Pair potential type");

      std::vector<double> global_coeffs;
      {
        const std::vector<std::string> tmp =
          dealii::Utilities::split_string_list(prm.get("Pair global coefficients"), ',');
        for ( const auto &c : tmp)
          global_coeffs.push_back(dealii::Utilities::string_to_double(c));
      }

      const std::vector<std::vector<std::string> > list_of_coeffs_per_type =
        Utilities::split_list_of_string_lists(prm.get("Pair specific coefficients"),';',',');

      if (pair_potential_type == "Coulomb Wolf")
        {
          AssertThrow (global_coeffs.size()==2,
                       ExcMessage("Invalid Pair global coefficients provided "
                                  "for the Pair potential type: "
                                  "Coulomb Wolf."));
          pair_potential =
            std::make_shared<Potential::PairCoulWolfManager> (global_coeffs[0],
                                                              global_coeffs[1]);
        }
      else if (pair_potential_type == "LJ")
        {
          AssertThrow (global_coeffs.size()==1,
                       ExcMessage("Invalid Pair global coefficients provided "
                                  "for the Pair potential type: LJ."));
          pair_potential =
            std::make_shared<Potential::PairLJCutManager> (global_coeffs[0]);

          for (const auto &specific_coeffs : list_of_coeffs_per_type)
            {
              // Pair specific coefficients = 0, 1, 2.5, 1.0
              // atom type 0 and 1 interact with epsilon = 2.5 and r_m = 1.
              AssertThrow (specific_coeffs.size() == 4,
                           ExcMessage("Only two specific coefficients should be "
                                      "provided the first element being "
                                      "epsilon and second being r_m as "));

              const types::atom_type
              i = dealii::Utilities::string_to_int(specific_coeffs[0]),
              j = dealii::Utilities::string_to_int(specific_coeffs[1]);

              std::vector<double> coeffs(2);
              coeffs[0] = dealii::Utilities::string_to_double(specific_coeffs[2]);
              coeffs[1] = dealii::Utilities::string_to_double(specific_coeffs[3]);

              (std::static_pointer_cast<Potential::PairLJCutManager>
               (pair_potential))->declare_interactions (i,
                                                        j,
                                                        Potential::InteractionTypes::LJ,
                                                        coeffs);
            }
        }
      else
        {
          AssertThrow (false, ExcInternalError());
        }

    }
    prm.leave_subsection();
    prm.enter_subsection("Configure QC");
    {
      //TODO: Max->Maximum
      maximum_search_radius = prm.get_double("Max search radius");
      cluster_radius = prm.get_double( "Cluster radius");
    }
    prm.leave_subsection();
  }

}
