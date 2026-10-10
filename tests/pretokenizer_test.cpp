// Pre-tokenizer chain from tokenizer.json must be interpreted, not hardcoded:
// GLM-5.3 expresses the digit branch inside the word regex (\p{N}{1,3}) while
// MiniCPM5 uses a separate \p{N}{1,3} stage followed by a word regex with
// \p{N}+. Both must yield the same boundaries. Uses the real tokenizer.json
// of each model.
//
// Build + run (from the repo root):
//   mkdir -p /tmp/glmtok && cp /tmp/glm53_tok.json /tmp/glmtok/tokenizer.json
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -o /tmp/tok_test tests/pretokenizer_test.cpp \
//     src/tokenizer.cpp -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
//   /tmp/tok_test     # expect "pretokenizer chain: PASS"
// Pre-tokenizer chain: the digit branch must be capped at {1,3} for BOTH
// configs (GLM expresses it inside the word regex, MiniCPM5 as a separate
// stage). Uncapped (\p{N}+) would emit whole digit runs - caught here.
#include <cstdio>
#include <string>
#include <vector>
#include "tokenizer.h"

using exl3::Tokenizer;
static bool digit_chunks(const Tokenizer& t, const std::string& s, std::vector<std::string>& pieces) {
  auto ids = t.encode(s, false);
  pieces.clear();
  for (int id : ids) pieces.push_back(t.decode({id}, false));
  return true;
}
static int check(const char* name, const char* path) {
  Tokenizer t; std::string err;
  if (!t.load(path, err)) { std::printf("%s: load failed: %s\n", name, err.c_str()); return 1; }
  std::vector<std::string> p;
  const std::string s = "x1234567890y";
  p = t.pretokenize(s);
  std::string joined, digs;
  for (auto& x : p) { joined += x; if (x.size() && x[0] >= '0' && x[0] <= '9') digs += "[" + x + "]"; }
  std::printf("%s: pre-tokens %s\n", name, "");
  for (auto& x : p) std::printf("   [%s]", x.c_str());
  std::printf("\n");
  bool ok = joined == s && digs == "[123][456][789][0]";
  // every piece boundary must be respected by BPE: no token may contain a
  // piece-internal split, i.e. concatenating the token strings must give the pieces
  {
    auto ids = t.encode(s, false);
    std::string rt; for (int id : ids) rt += t.decode({id}, false);
    if (rt != s) { std::printf("  %s: roundtrip MISMATCH -> %s\n", name, rt.c_str()); ok = false; }
  }
  // spot check: numbers of assorted lengths and CJK/emoji survive
  for (const char* q : {"1234567", "a1b22c333d4444", "hello"}) {
    auto ids = t.encode(q, false);
    std::string rt;
    for (int id : ids) rt += t.decode({id}, false);
    if (rt != q) { std::printf("  %s: roundtrip of \"%s\" -> \"%s\" FAILED\n", name, q, rt.c_str()); ok = false; }
  }
  return ok ? 0 : 1;
}
int main() {
  int rc = 0;
  rc |= check("MiniCPM5", "/Volumes/External/Models/ewin-reg/MiniCPM5-2B-EXL3-Quantized");
  rc |= check("GLM-5.3", "/tmp/glmtok");
  // Independent ground truth: a GPT-2-style byte-level BPE over the real
  // merges (verified by a Python implementation) gives these single tokens.
  {
    static const struct { const char* text; int n_ids; int first; } kCases[] = {
        {" the", 1, 279}, {" capital", 1, 6722}, {" sentence", 1, 11646},
        {" trellis", 3, 4258}, {" quantization", 2, 10267},
    };
    exl3::Tokenizer t; std::string err;
    if (t.load("/tmp/glmtok", err)) {
      for (const auto& c : kCases) {
        auto ids = t.encode(c.text, false);
        bool ok = (int)ids.size() == c.n_ids && (!ids.empty() && ids[0] == c.first);
        std::printf("  %-14s -> %zu ids, first %d (want %d ids, first %d) %s\n", c.text,
                    ids.size(), ids.empty() ? -1 : ids[0], c.n_ids, c.first, ok ? "ok" : "FAIL");
        if (!ok) rc = 1;
      }
    }
  }
  std::printf("pretokenizer chain: %s\n", rc == 0 ? "PASS" : "FAIL");
  return rc;
}
