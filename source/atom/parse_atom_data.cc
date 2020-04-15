
#include <deal.II-qc/atom/parse_atom_data.h>

#include <algorithm>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

template <int spacedim, int atomicity>
ParseAtomData<spacedim, atomicity>::ParseAtomData()
  : n_atoms(0)
  , n_bonds(0)
  , n_atom_types(0)
  , line_no(0)
{}

template <int spacedim, int atomicity>
void
ParseAtomData<spacedim, atomicity>::parse(
  std::istream &                              is,
  std::vector<Molecule<spacedim, atomicity>> &molecules,
  std::vector<types::charge> &                charges,
  std::vector<double> &                       masses,
  types::bond_type (&bonds)[atomicity][atomicity])
{
  AssertThrow(is, ExcIO());

  std::string line;

  // Some temporary variables
  unsigned int nangles    = 0;
  unsigned int ndihedrals = 0;
  unsigned int nimpropers = 0;

  // Dimensions of the simulation box
  double xlo(0.), xhi(0.), ylo(0.), yhi(0.), zlo(0.), zhi(0.);

  // Tilts of the simulation box
  double xy(0.), xz(0.), yz(0.);

  // Read comment line (first line)
  line_no++;
  if (!std::getline(is, line))
    Assert(false, ExcReadFailed(line_no));

  // Read main header declarations
  while (std::getline(is, line))
    {
      line_no++;

      // strip all the comments and all white characters
      line = strip(line);

      // Skip empty lines
      if (line.size() == 0)
        continue;

      // Read and store n_atoms, n_bonds, ...
      if (line.find("atoms") != std::string::npos)
        {
          unsigned long long int n_atoms_tmp;
          if (sscanf(line.c_str(), "%llu", &n_atoms_tmp) != 1)
            AssertThrow(false, ExcInvalidValue(line_no, "atoms"));
          if (n_atoms_tmp <= UINT_MAX)
            n_atoms = static_cast<types::global_atom_index>(n_atoms_tmp);
          else
            AssertThrow(false,
                        ExcMessage("The number of atoms specified is "
                                   "greater than the maximum finite value of "
                                   "`types::global_atom_type`!"
                                   "Try building deal.II with 64bit"
                                   "index space."));
          Assert(n_atoms % atomicity == 0,
                 ExcMessage("The total number of atoms provided in the "
                            "atom data do not form integer molecules. "));
        }
      else if (line.find("bonds") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%u", &n_bonds) != 1)
            Assert(false, ExcInvalidValue(line_no, "bonds"));
          // Stringent checks performed above for n_atoms are not performed.
          Assert(n_bonds >= 0,
                 ExcMessage("Invalid number of bonds in the atom data!"))
        }
      else if (line.find("angles") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%u", &nangles) != 1)
            Assert(false, ExcInvalidValue(line_no, "angles"));
          Assert(nangles == 0, ExcIrrelevant(line_no));
        }
      else if (line.find("dihedrals") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%u", &ndihedrals) != 1)
            Assert(false, ExcInvalidValue(line_no, "dihedrals"));
          Assert(ndihedrals == 0, ExcIrrelevant(line_no));
        }
      else if (line.find("impropers") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%u", &nimpropers) != 1)
            Assert(false, ExcInvalidValue(line_no, "impropers"));
          Assert(nimpropers == 0, ExcIrrelevant(line_no));
        }
      else if (line.find("atom types") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%zu", &n_atom_types) != 1)
            Assert(false, ExcInvalidValue(line_no, "number of atom types"));
        }
      else if (line.find("bond types") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%zu", &n_bond_types) != 1)
            Assert(false, ExcInvalidValue(line_no, "number of bond types"));
        }
      else if (line.find("angle types") != std::string::npos)
        {
          size_t n_angle_types;
          if (sscanf(line.c_str(), "%zu", &n_angle_types) != 1)
            Assert(false, ExcInvalidValue(line_no, "number of bond types"));
        }
      else if (line.find("dihedral types") != std::string::npos)
        {
          size_t n_dihedral_types;
          if (sscanf(line.c_str(), "%zu", &n_dihedral_types) != 1)
            Assert(false, ExcInvalidValue(line_no, "number of bond types"));
        }
      else if (line.find("xlo xhi") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%lf %lf", &xlo, &xhi) != 2)
            Assert(false,
                   ExcInvalidValue(line_no, "simulation box dimensions"));
        }
      else if (line.find("ylo yhi") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%lf %lf", &ylo, &yhi) != 2)
            Assert(false,
                   ExcInvalidValue(line_no, "simulation box dimensions"));
        }
      else if (line.find("zlo zhi") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%lf %lf", &zlo, &zhi) != 2)
            Assert(false,
                   ExcInvalidValue(line_no, "simulation box dimensions"));
        }
      else if (line.find("xy xz yz") != std::string::npos)
        {
          if (sscanf(line.c_str(), "%lf %lf %lf", &xy, &xz, &yz) != 3)
            Assert(false, ExcInvalidValue(line_no, "simulation box tilts"));
          Assert((((xy == xz) == yz) == 0), ExcIrrelevant(line_no));
        }
      else
        break;
    }

  Assert(xhi >= xlo || yhi >= ylo || zhi >= zlo,
         ExcMessage("Invalid simulation box size"));

  // TODO: Extra Asserts with dim and simulation box dimensions
  // TODO: copy simulation box dimensions, check for inconsistent mesh?

  molecules.resize(n_atoms / atomicity);
  charges.resize(n_atom_types);
  masses.resize(n_atom_types, 0.);

  // Initialize all the bonds to invalid entries.
  for (auto i = 0; i < atomicity; ++i)
    for (auto j = 0; j < atomicity; ++j)
      bonds[i][j] = numbers::invalid_bond_value;

  // line has been read by previous while loop but wasn't parsed for keyword
  // sections
  --line_no;
  do
    {
      // Read sections with keyword headings (Atoms, Masses)
      // For now ignoring other keyword headings // TODO?
      if (dealii::Utilities::match_at_string_start(line, "Masses"))
        {
          // parse_masses is allowed to be executed multiple times.
          parse_masses(is, masses);
        }
      else if (dealii::Utilities::match_at_string_start(line, "Atoms"))
        {
          // parse_atoms is allowed to be executed multiple times.
          parse_atoms(is, molecules, charges);
        }
      else if (dealii::Utilities::match_at_string_start(line, "Bonds"))
        {
          // parse_bonds is allowed to be executed multiple times.
          parse_bonds(is, bonds);
        }
      ++line_no;
    }
  while (std::getline(is, line));

  return;
}



template <int spacedim, int atomicity>
void
ParseAtomData<spacedim, atomicity>::parse_atoms(
  std::istream &                              is,
  std::vector<Molecule<spacedim, atomicity>> &molecules,
  std::vector<types::charge> &                charges)
{
  std::string line;

  // Atom counting starts from 1 in the atom data and
  // we count from 0.
  unsigned long long int i_atom_index;

  // Molecule count begins from 1 in the atom data and
  // we count from with 0.
  // i_molecule = i_molecule_index-1
  unsigned long long int   i_molecule_index;
  types::global_atom_index i_molecule;

  // i_atom_type = i_atom_char_type -1
  // atom type count begins from 1 in atom data and
  // we count from 0.
  unsigned char    i_atom_char_type;
  types::atom_type i_atom_type;

  // Temporary variable to count number of atom types.
  std::vector<types::atom_type> unique_types;

  // Charges read for each atom.
  double i_q;

  // Prepare position of the atom in this container.
  std::array<double, 3> position;

  // Since we cannot push_back atoms into molecules, we need to keep track
  // of the number of atoms already inserted into molecules.
  std::vector<int> n_atoms_added_per_molecule(n_atoms / atomicity, 0);

  // Prepare an atom in this container for each line read.
  Atom<spacedim> temporary_atom;

  // Store atom attributes and positions.
  types::global_atom_index i = 0;
  while (std::getline(is, line) && i < n_atoms)
    {
      ++line_no;
      line = strip(line);
      if (line.size() == 0)
        continue;

      // Not reading molecular_id
      if (sscanf(line.c_str(),
                 "%llu  %llu " UC_SCANF_STR " %lf %lf %lf %lf",
                 &i_atom_index,
                 &i_molecule_index,
                 &i_atom_char_type,
                 &i_q,
                 &position[0],
                 &position[1],
                 &position[2]) != 7)
        AssertThrow(false,
                    ExcInvalidValue(
                      line_no, "atom attributes under Atom keyword section"));

      Assert(i_atom_index <= n_atoms && i_atom_index > 0,
             ExcInvalidValue(line_no, "atom index (> number of atoms or <=0"));

      temporary_atom.global_index =
        static_cast<types::global_atom_index>(i_atom_index) - 1;

      i_atom_type         = static_cast<types::atom_type>(i_atom_char_type) - 1;
      temporary_atom.type = i_atom_type;

      Assert(i_atom_type >= 0 && i_atom_type < 256,
             ExcInvalidValue(line_no, "atom type attribute"));

      // TODO: 1 and 2 dim cases could be potentially incorrect
      // Possible way to correct this is to ask user axes dimensionality
      Assert(spacedim <= 3, ExcNotImplemented());

      for (int d = 0; d < spacedim; ++d)
        temporary_atom.position[d] = position[d];

      temporary_atom.initial_position = temporary_atom.position;

      //---Atom attributes are prepared for temporary_atom.

      //---Prepare charges.

      if (std::find(unique_types.begin(), unique_types.end(), i_atom_type) ==
          unique_types.end())
        {
          unique_types.push_back(i_atom_type);
          charges[i_atom_type] = static_cast<types::charge>(i_q);
        }
      else
        Assert(charges[i_atom_type] == static_cast<types::charge>(i_q),
               ExcInvalidValue(line_no, "charge attribute"));

      //---Ready to insert atom into molecule.

      i_molecule = static_cast<types::global_atom_index>(i_molecule_index) - 1;

      // Set molecule global id
      molecules[i_molecule].global_index = i_molecule;

      // Add atom to the molecule.
      molecules[i_molecule].atoms[n_atoms_added_per_molecule[i_molecule]] =
        temporary_atom;

      // Increment the number of atoms added to ith molecule.
      n_atoms_added_per_molecule[i_molecule]++;

      // Set some member variables of i_molecule to invalid values as
      // the current class cannot initialize to correct values.
      molecules[i_molecule].local_index = dealii::numbers::invalid_unsigned_int;
      molecules[i_molecule].cluster_weight = numbers::invalid_cluster_weight;

      ++i;
    }

  Assert(unique_types.size() == n_atom_types, ExcInternalError());

  Assert(i == n_atoms,
         ExcMessage("The number of atoms do not match the number of entries "
                    "under Atoms keyword section"));

#ifdef DEBUG

  // Make sure that for each molecule atomicity-number of atoms are read from
  // atom data. If this is not the case then one or more of the atoms of
  // molecules are not set.
  for (const auto &entry : n_atoms_added_per_molecule)
    Assert(entry == atomicity,
           ExcMessage("The number of atoms parsed for a molecule is not "
                      "equal to its atomicity."));
#endif

  //---Now sort atoms within molecules according to their global_atom_index.

  // Making a lambda for atom stamp comparision
  auto comparator_atom_type = [](const decltype(temporary_atom) &a,
                                 const decltype(temporary_atom) &b) {
    return a.global_index < b.global_index;
  };

  for (auto &molecule : molecules)
    {
      std::sort(molecule.atoms.begin(),
                molecule.atoms.end(),
                comparator_atom_type);
    }

  return;
}



template <int spacedim, int atomicity>
void
ParseAtomData<spacedim, atomicity>::parse_masses(std::istream &       is,
                                                 std::vector<double> &masses)
{
  std::string line;
  size_t      i_atom_type;
  double      i_mass;

  size_t i = 0;
  // Store atom masses for different atom types
  while (std::getline(is, line) && i < n_atom_types)
    {
      ++line_no;
      line = strip(line);
      if (line.size() == 0)
        continue;

      if (sscanf(line.c_str(), "%zu %lf", &i_atom_type, &i_mass) != 2)
        AssertThrow(false, ExcInvalidValue(line_no, "Mass"));

      Assert(i_atom_type <= n_atom_types && i_atom_type > 0,
             ExcInvalidValue(line_no,
                             "atom type index (> number of atom types "
                             "or <= 0)"));

      // Copy masses for all atom types
      masses[i_atom_type - 1] = i_mass;
      ++i;
    }
  line_no++;
  Assert(i == masses.size(),
         ExcMessage("The number of different atom types "
                    "do not match the number of entries "
                    "under Masses keyword section"));
  return;
}



template <int spacedim, int atomicity>
void
ParseAtomData<spacedim, atomicity>::parse_bonds(
  std::istream &is,
  types::bond_type (&bonds)[atomicity][atomicity])
{
  // Store bond information in this container.
  std::vector<std::tuple<types::global_atom_index,
                         types::global_atom_index,
                         types::bond_type>>
    bonds_vec(n_bonds);

  std::string line;

  // Atom counting starts from 1 in the atom data and
  // we count from 0.
  unsigned long long int i_atom_index, j_atom_index;

  // Store bond attributes in the following variables.
  unsigned long long int ij_bond;
  types::bond_type       ij_bond_type;

  types::global_bond_index i = 0;
  while (std::getline(is, line) && i < n_bonds)
    {
      ++line_no;
      line = strip(line);
      if (line.size() == 0)
        continue;

      if (sscanf(line.c_str(),
                 "%llu  " UC_SCANF_STR " %llu %llu",
                 &ij_bond,
                 &ij_bond_type,
                 &i_atom_index,
                 &j_atom_index) != 4)
        AssertThrow(false,
                    ExcInvalidValue(
                      line_no, "bond attributes under Bonds keyword section"));

      bonds_vec[i++] =
        std::make_tuple(i_atom_index - 1, j_atom_index - 1, ij_bond_type);
    }

  line_no++;
  Assert(i == bonds_vec.size(),
         ExcMessage("The number of different atom types "
                    "do not match the number of entries "
                    "under Masses keyword section"));

  // In most of the applications, we typically use a single Molecule type.
  // This results in bond information duplication.
  // In the following, an assumption is made that the entire information
  // regarding bonds can be condensed by looking at the bonds inside
  // a single Molecule.
  // Note: The notion of Molecule here is different from that of LAMMPS
  // For example, in LAMMPS, a core and a shell particle of Sodium in NaCl
  // constitutes a molecule as is given molecule ID accordinly.
  // However, here the complete unit cell of NaCl would be considered as
  // a Molecule.

  // We just need to parse only the first atomicity/2 number of bonds.
  for (auto i = 0; i < atomicity / 2; ++i)
    {
      types::global_atom_index i_atom, j_atom;
      types::bond_type         ij_bond;

      std::tie(i_atom, j_atom, ij_bond) = bonds_vec[i];
      auto &ij_entry                    = bonds[i_atom][j_atom];
      auto &ji_entry                    = bonds[j_atom][i_atom];

      ij_entry = (ij_entry == numbers::invalid_bond_value) ? ij_bond : ij_entry;

      ji_entry = (ji_entry == numbers::invalid_bond_value) ? ij_bond : ji_entry;
    }

  return;
}

template <int spacedim, int atomicity>
std::string
ParseAtomData<spacedim, atomicity>::strip(const std::string &input)
{
  std::string line(input);

  // Find if # is present
  size_t trim_pos = line.find("#");

  // Trim all the stuff after #
  if (trim_pos != std::string::npos)
    line.resize(trim_pos);

  // Trim line from left and right with " \t\n\r"
  line = dealii::Utilities::trim(line);

  return line;
}


#define PARSE_ATOM_DATA(R, X)                               \
  template class ParseAtomData<FIRST_OF_TWO_IS_SPACEDIM(X), \
                               SECOND_OF_TWO_IS_ATOMICITY(X)>;

INSTANTIATE_WITH_SPACEDIM_AND_ATOMICITY(R, PARSE_ATOM_DATA)


DEAL_II_QC_NAMESPACE_CLOSE
