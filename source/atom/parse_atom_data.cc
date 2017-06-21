
#include <algorithm>

#include <deal.II-qc/atom/parse_atom_data.h>

namespace dealiiqc
{
  using namespace dealii;

  template<int spacedim, int atomicity>
  ParseAtomData<spacedim, atomicity>::ParseAtomData()
    :
    n_atoms(0),
    n_atom_types(0),
    line_no(0)
  {}

  template<int spacedim, int atomicity>
  void
  ParseAtomData<spacedim, atomicity>::parse (std::istream                               &is,
                                             std::vector<Molecule<spacedim, atomicity>> &molecules,
                                             std::vector<types::charge>                 &charges,
                                             std::vector<double>                        &masses)
  {
    AssertThrow (is, ExcIO());

    std::string line;

    // Some temporary variables
    unsigned int nbonds = 0;
    unsigned int nangles = 0;
    unsigned int ndihedrals = 0;
    unsigned int nimpropers = 0;

    // Dimensions of the simulation box
    double xlo(0.),xhi(0.),
           ylo(0.),yhi(0.),
           zlo(0.),zhi(0.);

    // Tilts of the simulation box
    double xy(0.),xz(0.),yz(0.);

    // Read comment line (first line)
    line_no++;
    if (!std::getline(is,line))
      Assert( false, ExcReadFailed(line_no));

    // Read main header declarations
    while ( std::getline(is,line) )
      {
        line_no++;

        // strip all the comments and all white characters
        line = strip(line);

        // Skip empty lines
        if ( line.size()==0)
          continue;

        // Read and store n_atoms, nbonds, ...
        if (line.find("atoms") != std::string::npos)
          {
            unsigned long long int n_atoms_tmp;
            if (sscanf(line.c_str(), "%llu", &n_atoms_tmp) != 1)
              AssertThrow( false, ExcInvalidValue(line_no,"atoms"));
            if ( n_atoms_tmp <= UINT_MAX )
              n_atoms = static_cast<types::global_atom_index>(n_atoms_tmp);
            else
              AssertThrow( false,
                           ExcMessage("The number of atoms specified "
                                      "is more than what `typedefs::global_atom_index` can work with "
                                      "try building deal.II with 64bit index space"));
            Assert (n_atoms % atomicity == 0,
                    ExcMessage("The total number of atoms provided in the "
                               "atom data do not form integer molecules. "));
          }
        else if (line.find("bonds") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nbonds) != 1)
              Assert( false, ExcInvalidValue(line_no,"bonds"));
            Assert( nbonds==0, ExcIrrelevant(line_no));
          }
        else if (line.find("angles") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nangles) != 1)
              Assert( false, ExcInvalidValue(line_no,"angles"));
            Assert( nangles==0, ExcIrrelevant(line_no));
          }
        else if (line.find("dihedrals") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &ndihedrals) != 1)
              Assert( false, ExcInvalidValue(line_no,"dihedrals"));
            Assert( ndihedrals==0, ExcIrrelevant(line_no));
          }
        else if (line.find("impropers") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%u", &nimpropers) != 1)
              Assert( false, ExcInvalidValue(line_no,"impropers"));
            Assert( nimpropers==0, ExcIrrelevant(line_no));
          }
        else if (line.find("atom types") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%zu", &n_atom_types) != 1)
              Assert( false , ExcInvalidValue(line_no,"number of atom types"));
          }
        else if (line.find("xlo xhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &xlo, &xhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("ylo yhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &ylo, &yhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("zlo zhi") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf", &zlo, &zhi) != 2)
              Assert( false, ExcInvalidValue(line_no,"simulation box dimensions"));
          }
        else if (line.find("xy xz yz") != std::string::npos)
          {
            if (sscanf(line.c_str(), "%lf %lf %lf", &xy, &xz, &yz) != 3)
              Assert( false, ExcInvalidValue(line_no,"simulation box tilts"));
            Assert((((xy==xz)==yz)==0), ExcIrrelevant(line_no));
          }
        else break;
      }

    Assert( xhi >= xlo || yhi >= ylo || zhi >= zlo,
            ExcMessage("Invalid simulation box size"))
    // TODO: Extra Asserts with dim and simulation box dimensions
    // TODO: copy simulation box dimensions, check for inconsistent mesh?

    molecules.resize(n_atoms/atomicity);
    charges.resize(n_atom_types);
    masses.resize(n_atom_types, 0.);

    // line has been read by previous while loop but wasn't parsed for keyword sections
    --line_no;
    do
      {
        // Read sections with keyword headings (Atoms, Masses)
        // For now ignoring other keyword headings // TODO?
        if ( dealii::Utilities::match_at_string_start(line, "Masses") )
          {
            // parse_masses is allowed to be executed multiple times.
            parse_masses(is, masses);
          }
        else if ( dealii::Utilities::match_at_string_start(line,"Atoms"))
          {
            // parse_atoms is allowed to be executed multiple times.
            parse_atoms(is, molecules, charges);
          }
        ++line_no;
      }
    while ( std::getline(is,line) );

    return;
  }



  template<int spacedim, int atomicity>
  void
  ParseAtomData<spacedim, atomicity>::parse_atoms (std::istream                               &is,
                                                   std::vector<Molecule<spacedim, atomicity>> &molecules,
                                                   std::vector<types::charge>                 &charges)
  {
    std::string line;

    // Atom counting starts from 1 in the atom data and
    // we count from 0.
    unsigned long long int  i_atom_index;

    // Molecule count begins from 1 in the atom data and
    // we count from with 0.
    // i_molecule = i_molecule_index-1
    unsigned long long int i_molecule_index;
    types::global_atom_index i_molecule;

    // i_atom_type = i_atom_char_type -1
    // atom type count begins from 1 in atom data and
    // we count from 0.
    unsigned char i_atom_char_type;
    types::atom_type i_atom_type;

    // Temporary variable to count number of atom types.
    std::vector<types::atom_type> unique_types;

    // Charges and positions read for each atom.
    double i_q, i_x, i_y, i_z;

    // Since we cannot push_back atoms into molecules, we need to keep track
    // of the number of atoms already inserted into molecules.
    std::vector<int> n_atoms_added_per_molecule(n_atoms/atomicity, 0);

    // Prepare an atom in this container for each line read.
    Atom<spacedim> temporary_atom;

    // Store atom attributes and positions.
    types::global_atom_index i=0;
    while ( std::getline(is,line) && i < n_atoms)
      {
        ++line_no;
        line = strip(line);
        if ( line.size()==0)
          continue;

        // Not reading molecular_id
        if (sscanf(line.c_str(), "%llu  %llu " UC_SCANF_STR " %lf %lf %lf %lf",
                   &i_atom_index, &i_molecule_index, &i_atom_char_type, &i_q, &i_x, &i_y, &i_z ) !=7)
          AssertThrow (false,
                       ExcInvalidValue( line_no,
                                        "atom attributes under Atom keyword section"));

        Assert (i_atom_index<=n_atoms && i_atom_index>0,
                ExcInvalidValue( line_no,
                                 "atom index (> number of atoms or <=0"));

        temporary_atom.global_index =
          static_cast<types::global_atom_index>(i_atom_index) -1;

        i_atom_type =
          temporary_atom.type =
            static_cast<types::atom_type> (i_atom_char_type) -1;

        Assert (i_atom_type >= 0 && i_atom_type < 256,
                ExcInvalidValue(line_no, "atom type attribute"));

        // TODO: 1 and 2 dim cases could be potentially incorrect
        // Possible way to correct this is to ask user axes dimensionality
        if (spacedim==1)
          {
            temporary_atom.position[0]         = i_x;
            temporary_atom.initial_position[0] = i_x;
          }
        else if (spacedim==2)
          {
            temporary_atom.position[0]         = i_x;
            temporary_atom.position[1]         = i_y;
            temporary_atom.initial_position[0] = i_x;
            temporary_atom.initial_position[1] = i_y;
          }
        else if (spacedim==3)
          {
            temporary_atom.position[0]         = i_x;
            temporary_atom.position[1]         = i_y;
            temporary_atom.position[2]         = i_z;
            temporary_atom.initial_position[0] = i_x;
            temporary_atom.initial_position[1] = i_y;
            temporary_atom.initial_position[2] = i_z;
          }

        //---Atom attributes are prepared for temporary_atom.

        //---Prepare chagres.

        if (std::find(unique_types.begin(), unique_types.end(), i_atom_type) == unique_types.end())
          {
            unique_types.push_back(i_atom_type);
            charges[i_atom_type] = static_cast<types::charge>(i_q);
          }
        else
          Assert (charges[i_atom_type]==static_cast<types::charge>(i_q),
                  ExcInvalidValue(line_no, "charge attribute"));

        //---Ready to insert atom to into molecule.

        i_molecule = static_cast<types::global_atom_index>(i_molecule_index) -1;

        // Add atom to the molecule.
        molecules[i_molecule].atoms[n_atoms_added_per_molecule[i_molecule]] =
          temporary_atom;

        // Increment the number of atoms added to ith molecule.
        n_atoms_added_per_molecule[i_molecule]++;

        // Set some member variables of i_molecule to invalid values as
        // the current class cannot initialize to correct values.
        molecules[i_molecule].local_index    = dealii::numbers::invalid_unsigned_int;
        molecules[i_molecule].cluster_weight = numbers::invalid_cluster_weight;

        ++i;
      }

    Assert (unique_types.size()==n_atom_types,
            ExcInternalError());

    Assert (i==n_atoms,
            ExcMessage("The number of atoms do not match the number of entries "
                       "under Atoms keyword section"));

    //---At this point atoms in molecules are not according to their stamps.
    //   For now it is not possible to ensure correct order of stamps for
    //   repeated atom types.
    //   Example: A2B molecule could have two possible stamp orderings are
    //            possible.
    //
    //   case 1:  0 1 2         ----------> stamp order
    //            0 0 1         ----------> type order
    //
    //   case 2:  1 0 2         ----------> stamp order
    //            0 0 1         ----------> type order

    // Making a lambda for atom type comparision
    auto comparator_atom_type = [] (const decltype(temporary_atom) &a,
                                    const decltype(temporary_atom) &b)
    {
      return a.type < b.type;
    };

    for (auto &molecule : molecules)
      {
        std::sort (molecule.atoms.begin(),
                   molecule.atoms.end(),
                   comparator_atom_type);
      }

    return;
  }



  template<int spacedim, int atomicity>
  void
  ParseAtomData<spacedim, atomicity>::parse_masses (std::istream        &is,
                                                    std::vector<double> &masses)
  {
    std::string line;
    size_t i_atom_type;
    double i_mass;

    size_t i=0;
    // Store atom masses for different atom types
    while ( std::getline(is,line) && i<n_atom_types)
      {
        ++line_no;
        line = strip(line);
        if ( line.size()==0)
          continue;

        if (sscanf(line.c_str(), "%zu %lf", &i_atom_type, &i_mass) !=2)
          AssertThrow( false, ExcInvalidValue(line_no,"Mass"));

        Assert( i_atom_type <= n_atom_types && i_atom_type>0,
                ExcInvalidValue( line_no, "atom type index (> number of atom types "
                                 "or <= 0)"));

        // Copy masses for all atom types
        masses[i_atom_type-1] = i_mass;
        ++i;
      }
    line_no++;
    Assert (i==masses.size(),
            ExcMessage("The number of different atom types "
                       "do not match the number of entries "
                       "under Masses keyword section"));
    return;
  }



  template<int spacedim, int atomicity>
  std::string
  ParseAtomData<spacedim, atomicity>::strip (const std::string &input)
  {
    std::string line(input);

    // Find if # is present
    size_t trim_pos = line.find("#");

    // Trim all the stuff after #
    if ( trim_pos != std::string::npos )
      line.resize( trim_pos );

    // Trim line from left and right with " \t\n\r"
    line = dealii::Utilities::trim(line);

    return line;
  }


#define PARSE_ATOM_DATA(R, X) \
  template class ParseAtomData< FIRST_OF_TWO_IS_SAPCEDIM(X), \
                                SECOND_OF_TWO_IS_ATOMICITY(X)>;

  INSTANTIATE_WITH_SAPCEDIM_AND_ATOMICITY(R, PARSE_ATOM_DATA, (SAPCEDIM, ATOMICITY))

} /* namespace dealiiqc */
