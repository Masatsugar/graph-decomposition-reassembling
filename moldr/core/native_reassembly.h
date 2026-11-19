#pragma once

#include <GraphMol/RWMol.h>

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace moldr_native {
struct Result {
  // The first molecule with each SMILES is retained in atom insertion order.
  std::map<std::string, std::unique_ptr<RDKit::RWMol>> candidates;
  unsigned generated = 0;
  unsigned raw_unique = 0;
  unsigned invalid = 0;
};

// A missing result selects the existing adjacency implementation for special bonds.
std::optional<Result> generate(const RDKit::ROMol &base,
                               const RDKit::ROMol &fragment);
}  // namespace moldr_native
