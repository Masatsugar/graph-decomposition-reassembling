#include "native_reassembly.h"

#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <RDGeneral/Exceptions.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace moldr_native {
namespace {
using RDKit::Atom;
using RDKit::Bond;
using RDKit::ROMol;
using RDKit::RWMol;
using Index = unsigned;
using Ends = std::pair<Index, Index>;
using BondSpec = std::tuple<Index, Index, Bond::BondType>;

bool validate(const ROMol &mol) {
  if (!mol.getNumAtoms()) {
    throw std::invalid_argument("Reassembly requires a valid, nonempty molecule.");
  }
  constexpr auto stereo_error =
      "Stereochemical input is unsupported for adjacency-based reassembly; "
      "remove stereochemistry explicitly before use.";
  for (const auto *atom : mol.atoms()) {
    if (atom->getChiralTag() != Atom::CHI_UNSPECIFIED) {
      throw std::invalid_argument(stereo_error);
    }
  }
  bool supported = true;
  for (const auto *bond : mol.bonds()) {
    if (bond->getStereo() != Bond::STEREONONE || bond->getBondDir() != Bond::NONE) {
      throw std::invalid_argument(stereo_error);
    }
    const auto type = bond->getBondType();
    if (type != Bond::SINGLE && type != Bond::DOUBLE && type != Bond::TRIPLE &&
        type != Bond::AROMATIC) {
      supported = false;
    }
  }
  return supported;
}

struct AtomState {
  std::string symbol;
  unsigned implicit_valence, explicit_valence, total_valence;
};

struct Features {
  std::vector<AtomState> atoms;
  std::vector<Ends> ring_edges;

  explicit Features(const ROMol &input) {
    // RDKit may initialize ring information lazily. Preserve the caller's Mol.
    ROMol mol(input);
    for (const auto *atom : mol.atoms()) {
      atoms.push_back({atom->getSymbol(), atom->getValence(Atom::ValenceType::IMPLICIT),
                       atom->getValence(Atom::ValenceType::EXPLICIT),
                       atom->getTotalValence()});
    }
    if (mol.getNumBonds() && !mol.getRingInfo()->isSssrOrBetter()) {
      RDKit::MolOps::findSSSR(mol);
    }
    for (const auto *bond : mol.bonds()) {
      const auto a = bond->getBeginAtomIdx(), b = bond->getEndAtomIdx();
      if (mol.getRingInfo()->numBondRings(bond->getIdx()) &&
          atoms[a].implicit_valence && atoms[b].implicit_valence) {
        ring_edges.emplace_back(std::min(a, b), std::max(a, b));
      }
    }
    std::sort(ring_edges.begin(), ring_edges.end());
  }
};

struct Pair {
  RWMol atoms_only;
  std::vector<BondSpec> bonds;
  Result result;
  std::set<std::string> kind_keys;

  void append(const ROMol &mol) {
    const auto offset = atoms_only.getNumAtoms();
    for (const auto *atom : mol.atoms()) {
      const auto idx = atoms_only.addAtom(atom->copy(), false, true);
      atoms_only.getAtomWithIdx(idx)->setAtomMapNum(0);
    }
    for (const auto *bond : mol.bonds()) {
      const auto a = bond->getBeginAtomIdx() + offset;
      const auto b = bond->getEndAtomIdx() + offset;
      bonds.emplace_back(std::min(a, b), std::max(a, b), bond->getBondType());
    }
  }

  // For a node merge, the two endpoints in each pair coincide. Ring edges
  // always use increasing endpoint order, including the atom removal below.
  void attach(Ends keep, Ends drop) {
    auto redirect = [&](Index atom) {
      if (atom == drop.first) return keep.first;
      if (atom == drop.second) return keep.second;
      return atom;
    };
    std::vector<BondSpec> edited;
    edited.reserve(bonds.size());
    for (const auto &[a, b, type] : bonds) {
      if (a == drop.first && b == drop.second) continue;
      const auto x = redirect(a), y = redirect(b);
      edited.emplace_back(std::min(x, y), std::max(x, y), type);
    }
    // Match the upper-triangle bond insertion order of the Python builder.
    std::sort(edited.begin(), edited.end());
    auto mol = std::make_unique<RWMol>(atoms_only);
    for (const auto &[a, b, type] : edited) {
      mol->addBond(a, b, type);
    }
    if (drop.second != drop.first) mol->removeAtom(drop.second);
    mol->removeAtom(drop.first);

    ++result.generated;
    try {
      RDKit::MolOps::sanitizeMol(*mol);
      for (auto *bond : mol->bonds()) {
        if (bond->getIsAromatic()) bond->setBondType(Bond::AROMATIC);
      }
      const auto smiles = RDKit::MolToSmiles(*mol);
      kind_keys.insert(smiles);
      if (!result.candidates.count(smiles)) {
        // The Python wrapper preserves aromatic flags by default.
        RDKit::MolOps::Kekulize(*mol, false);
        result.candidates.emplace(smiles, std::move(mol));
      }
    } catch (const RDKit::MolSanitizeException &) {
      ++result.invalid;
    } catch (const IndexErrorException &) {
      throw;
    } catch (const KeyErrorException &) {
      throw;
    } catch (const std::runtime_error &) {
      // Includes RDKit's ValueErrorException and invariant violations.
      ++result.invalid;
    } catch (const std::invalid_argument &) {
      ++result.invalid;
    }
  }

  void finish_kind() {
    result.raw_unique += kind_keys.size();
    kind_keys.clear();
  }
};
}  // namespace

std::optional<Result> generate(const ROMol &base, const ROMol &fragment) {
  const bool base_supported = validate(base);
  const bool fragment_supported = validate(fragment);
  if (!base_supported || !fragment_supported) return std::nullopt;

  Features left(base), right(fragment);
  Pair pair;
  pair.append(base);
  pair.append(fragment);
  const auto offset = base.getNumAtoms();
  for (Index i = 0; i < left.atoms.size(); ++i) {
    const auto &a = left.atoms[i];
    for (Index j = 0; j < right.atoms.size(); ++j) {
      const auto &b = right.atoms[j];
      if (!a.implicit_valence || !b.implicit_valence ||
          a.explicit_valence + b.explicit_valence > std::max(a.total_valence, b.total_valence)) {
        continue;
      }
      Ends src{i, i}, dst{offset + j, offset + j};
      if (a.symbol != b.symbol) pair.attach(dst, src);
      pair.attach(src, dst);
    }
  }
  pair.finish_kind();

  for (const auto &[a, b] : left.ring_edges) {
    const auto labels = std::minmax(left.atoms[a].symbol, left.atoms[b].symbol);
    for (const auto &[c, d] : right.ring_edges) {
      Ends src{a, b}, dst{offset + c, offset + d};
      if (labels != std::minmax(right.atoms[c].symbol, right.atoms[d].symbol)) {
        pair.attach(dst, src);
      }
      pair.attach(src, dst);
    }
  }
  pair.finish_kind();
  return std::move(pair.result);
}
}  // namespace moldr_native
