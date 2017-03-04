
#include <dealiiqc/io/configure_qc.h>

namespace dealiiqc
{
  using namespace dealii;

  ConfigureQC::ConfigureQC( const std::string & parameter_filename)
  :
  mesh_file(std::string()), n_initial_global_refinements(1)
  {
    ParameterHandler prm;
    declare_parameters (prm);
    std::ifstream parameter_file (parameter_filename.c_str());
    if (!parameter_file)
    {
      parameter_file.close ();
      std::ostringstream message;
      message << "Input parameter file <"
	      << parameter_filename << "> not found."
	      << std::endl
	      << "Creating a template file of the same name."
	      << std::endl;
      std::ofstream parameter_out (parameter_filename.c_str());
      prm.print_parameters (parameter_out,
			    ParameterHandler::Text);
      AssertThrow (false, ExcMessage (message.str().c_str()));
    }
    const bool success = prm.read_input (parameter_file);
    AssertThrow (success, ExcMessage ("\nInvalid input parameter file.\n"));
    parse_parameters (prm);
  }

  std::string ConfigureQC::get_mesh_file () const
  {
    return mesh_file;
  }

  unsigned int ConfigureQC::get_n_initial_global_refinements() const
  {
    return n_initial_global_refinements;
  }

  void ConfigureQC::declare_parameters( ParameterHandler &prm )
  {
    // TODO: Write intput file name to the screen
    //deallog << std::endl << "Parsing qc input file " << filename << std::endl
    //	      << "for a " << dim << " dimensional simulation. " << std::endl;

    prm.declare_entry("Dimension", "2",
    		      Patterns::Integer(0),
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

    /* // TODO: Read atom infomration from LAMMPS like atom data file
    prm.enter_subsection ("Configure atoms");
    {
      // N atoms
      // a atom types
      //
      // Atoms
      // Atom_ID Atom_Type Atom_Charge Atom_X Atom_Y Atom_Z
    }
    prm.leave_subsection ();
    */

  }

  void ConfigureQC::parse_parameters( ParameterHandler &prm )
  {
    //const unsigned int dimension = prm.get_integer("Dimension");
    prm.enter_subsection("Configure mesh");
    {
      mesh_file                    = prm.get("Mesh file");
      n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
    }
    prm.leave_subsection();
  }

}
