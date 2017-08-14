
#ifndef __dealii_qc_compute_tools_h
#define __dealii_qc_compute_tools_h

#include <deal.II-qc/atom/molecule.h>
#include <deal.II-qc/potentials/potential_field.h>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A namespace for tools to compute energy and gradient among atoms
 * and molecules.
 */
namespace ComputeTools
{

  /**
   * Compute and return the energy and gradient values due to the presence of
   * @p atom with charge @p q (inside domain with @p material_id material id)
   * under external potential fields described through
   * @p external_potential_fields which is a mapping from material ids to
   * PotentialField functions.
   */
  template <int spacedim, bool ComputeGradient=true>
  inline
  std::pair<double, Tensor<1, spacedim> >
  energy_and_gradient
  (const std::multimap<unsigned int, std::shared_ptr<PotentialField<spacedim>>> &external_potential_fields,
   const dealii::types::material_id                                              material_id,
   const Atom<spacedim>                                                         &atom,
   const double                                                                 q)
  {
    // Prepare energy and gradient in the following variables.
    double external_energy = 0.;
    Tensor<1, spacedim> external_gradient;
    external_gradient = 0.;

    // Get the external potential field(s) for the provided material id.
    auto external_potential_fields_range =
      external_potential_fields.equal_range(material_id);

    for (auto
         potential_field  = external_potential_fields_range.first;
         potential_field != external_potential_fields_range.second;
         potential_field++)
      {
        external_energy  += potential_field->second->value (atom.position, q);

        if (ComputeGradient)
          external_gradient +=
            potential_field->second->gradient (atom.position, q);
      }

    return std::make_pair (external_energy, external_gradient);
  }



  /**
   * Compute and return the interaction energy and gradient values between two
   * atoms \f$i\f$ and \f$j\f$ located at \f$ \textbf x^i \f$ and
   * \f$ \textbf x^j \f$, respectively, interacting via a given @p potential.
   *
   * The potential energy can be computed using the empirical formula of
   * the given potential \f$ \phi^{}_{ij}(r^{ij}) \f$ where
   * \f$ r^{ij} = |\textbf r^{ij}| = |\textbf x^{i} - \textbf x^{j}| \f$ and
   * the gradient by
   * \f[
   *     \frac{\partial \phi_{ij}(r^{ij})}{\partial r^{ij}}
   *     \frac{\textbf r^{ij}}{r^{ij}}
   *     =
   *     {\phi}^\prime_{ij}(r^{ij})
   *     \frac{\textbf r^{ij}}{r^{ij}}.
   * \f].
   */
  template<typename PotentialType, int spacedim, bool ComputeGradient=true>
  inline
  std::pair<double, Tensor<1, spacedim> >
  energy_and_gradient (const PotentialType  &potential,
                       const Atom<spacedim> &atom_i,
                       const Atom<spacedim> &atom_j)
  {
    const Tensor<1, spacedim> rij = atom_i.position - atom_j.position;

    const double r_square = rij.norm_square();

    const std::pair<double, double> energy_and_gradient =
      potential.template energy_and_gradient<ComputeGradient> (atom_i.type,
                                                               atom_j.type,
                                                               r_square);
    return ComputeGradient
           ?
           std::make_pair (energy_and_gradient.first,
                           rij * (energy_and_gradient.second
                                  /
                                  std::sqrt(r_square)))
           :
           std::make_pair (energy_and_gradient.first,
                           rij * std::numeric_limits<double>::signaling_NaN());
  }



  /**
   * Compute and return the interaction energy and the table of gradient values
   * between two molecules \f$I\f$ and \f$J\f$ each consisting of
   * <tt>atomicity</tt>-number of atoms, interacting via a given
   * @p potential.
   *
   * The interaction energy of the two molecule \f$I\f$ and \f$J\f$ is given as
   * \f[
   *  E^{}_{IJ} =  \begin{cases}                   \,\,
   *                   \sum\limits_{i \, \in \, I}    \,\,
   *                   \sum\limits_{j \, \in \, J}    \,
   *                     \phi^{}_{ij}(r^{ij}),
   *                  & \text{if } \,  I \neq J \\[1em] \,\,
   *                    \sum\limits_{i \, \in \, I}    \,\,
   *                    \sum\limits_{J \, \ni \, j \, < \, i} \,
   *                      \phi^{}_{ij}(r^{ij}),
   *                  & \text{otherwise},
   *                \end{cases}
   * \f]
   * where
   * \f$ r^{ij} = |\textbf r^{ij}| = |\textbf x^{i}_{I} -\textbf x^{j}_{J}| \f$.
   * The table of gradient values consists of
   * <tt>atomicity</tt> \f$ \times \f$ <tt>atomicity</tt> number of entries. Each entry is a
   * Tensor of rank 1 and <tt>spacedim</tt>-number of indices. For a given pair
   * of molecules \f$I\f$ and \f$J\f$, the \f$(i,j)^{th}\f$ entry of the table
   * of gradient values is given as
   * \f[
   *     G_{IJ} \, (i,j) =  \frac{\partial \phi_{ij}(r^{ij})}
   *                             {\partial \textbf r^{ij}}
   *                     =  \frac{\partial \phi_{ij}(r^{ij})}{\partial r^{ij}}
   *                        \frac{\textbf r^{ij}}{r^{ij}}
   *                     =  {\phi}^\prime_{ij}(r^{ij})
   *                        \frac{\textbf r^{ij}}{r^{ij}}.
   * \f]
   */
  template<typename PotentialType, int spacedim, int atomicity=1, bool ComputeGradient=true>
  inline
  std::pair<double, Table<2, Tensor<1, spacedim> > >
  energy_and_gradient (const PotentialType                 &potential,
                       const Molecule<spacedim, atomicity> &molecule_I,
                       const Molecule<spacedim, atomicity> &molecule_J)
  {
    double energy = 0.;
    Table<2, Tensor<1, spacedim> > gradients (atomicity, atomicity);

    {
      Tensor<1, spacedim> temp;

      for (int i = 0; i < spacedim; ++i)
        temp[i] = ComputeGradient ?
                  0.              :
                  std::numeric_limits<double>::signaling_NaN();

      gradients.fill (temp);
    }

    const auto &atoms_I = molecule_I.atoms;
    const auto &atoms_J = molecule_J.atoms;

    if (molecule_I.global_index == molecule_J.global_index)
      // Intramolecular interaction.
      {
        for (unsigned int i = 0; i < atomicity; ++i)
          for (unsigned int j = 0; j < i; ++j)
            {
              const std::pair <double, Tensor<1, spacedim> >
              energy_and_gradient_tensor = energy_and_gradient
                                           <PotentialType, spacedim, ComputeGradient>(potential,
                                               atoms_I[i],
                                               atoms_I[j]);
              energy += energy_and_gradient_tensor.first;

              if (ComputeGradient)
                {
                  gradients(i,j) = energy_and_gradient_tensor.second;
                  gradients(j,i) = gradients(i,j);
                }
            }
      }
    else
      // Intermolecular interaction.
      {
        for (unsigned int i = 0; i < atomicity; ++i)
          for (unsigned int j = 0; j < atomicity; ++j)
            {
              const std::pair <double, Tensor<1, spacedim> >
              energy_and_gradient_tensor =
                energy_and_gradient
                <PotentialType, spacedim, ComputeGradient>(potential,
                                                           atoms_I[i],
                                                           atoms_J[j]);
              energy += energy_and_gradient_tensor.first;

              if (ComputeGradient)
                gradients(i,j) = energy_and_gradient_tensor.second;
            }
      }

    return std::make_pair (energy, gradients);
  }


} // namepsace ComputeTools


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_compute_tools_h */