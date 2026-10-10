#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

#include "nlohmann/json.hpp"
#include "unicode_data.h"
#include "util.h"

using nlohmann::json;

namespace exl3 {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Decode one code point at `pos`; returns the number of bytes consumed (>=1).
static int utf8_decode(const std::string& s, size_t pos, uint32_t& cp) {
  const unsigned char* p = (const unsigned char*)s.data() + pos;
  const size_t n = s.size() - pos;
  if (n == 0) { cp = 0; return 0; }
  unsigned char c = p[0];
  if (c < 0x80) { cp = c; return 1; }
  auto cont = [&](size_t i) { return i < n && (p[i] & 0xC0) == 0x80; };
  if ((c & 0xE0) == 0xC0 && cont(1)) {
    cp = ((uint32_t)(c & 0x1F) << 6) | (p[1] & 0x3F);
    return cp >= 0x80 ? 2 : 1; // reject overlong
  }
  if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
    cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    return cp >= 0x800 ? 3 : 1;
  }
  if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
    cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
         ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    return (cp >= 0x10000 && cp <= 0x10FFFF) ? 4 : 1;
  }
  cp = c;
  return 1;
}

static void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back((char)cp);
  } else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// Length of the longest prefix that is complete UTF-8 (valid, or invalid in a
// way that cannot be extended by more bytes). `lossy` then converts everything.
static size_t utf8_complete_prefix(const std::string& s) {
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = (unsigned char)s[i];
    size_t len = 1;
    if (c < 0x80) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    else { i += 1; continue; } // stray continuation byte: never completes
    if (i + len > n) break;    // incomplete tail
    uint32_t cp;
    int got = utf8_decode(s, i, cp);
    if (got == (int)len && (len == 1 || cp >= (len == 2 ? 0x80u : len == 3 ? 0x800u : 0x10000u))) {
      i += len;
    } else {
      i += 1; // invalid sequence: consume one byte
    }
  }
  return i;
}

static std::string utf8_lossy(const std::string& s, size_t upto) {
  std::string out;
  out.reserve(upto);
  size_t i = 0;
  while (i < upto) {
    uint32_t cp;
    int len = utf8_decode(s, i, cp);
    bool valid = false;
    if (len == 1) valid = cp < 0x80;
    else if (len == 2) valid = cp >= 0x80;
    else if (len == 3) valid = cp >= 0x800;
    else if (len == 4) valid = cp >= 0x10000 && cp <= 0x10FFFF;
    if (valid) {
      utf8_append(out, cp);
      i += len;
    } else {
      utf8_append(out, 0xFFFD);
      i += len > 0 ? len : 1;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Unicode properties
// ---------------------------------------------------------------------------

static bool in_ranges(const uint32_t (*tbl)[2], size_t count, uint32_t cp) {
  size_t lo = 0, hi = count;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < tbl[mid][0]) hi = mid;
    else if (cp > tbl[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

static inline bool cp_is_letter(uint32_t cp) {
  if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
  return in_ranges(uni::kLetterRanges, uni::kLetterRanges_count, cp);
}

static inline bool cp_is_number(uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(uni::kNumberRanges, uni::kNumberRanges_count, cp);
}

static inline bool cp_is_space(uint32_t cp) {
  switch (cp) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
    case 0x202F: case 0x205F: case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level alphabet
// ---------------------------------------------------------------------------

static const std::vector<uint32_t>& byte_to_cp_table() {
  static const std::vector<uint32_t> table = [] {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }
    std::vector<uint32_t> out(256);
    for (size_t i = 0; i < bs.size(); ++i) out[(size_t)bs[i]] = (uint32_t)cs[i];
    return out;
  }();
  return table;
}

static const std::unordered_map<uint32_t, uint8_t>& cp_to_byte_table() {
  static const std::unordered_map<uint32_t, uint8_t> table = [] {
    std::unordered_map<uint32_t, uint8_t> m;
    const auto& t = byte_to_cp_table();
    for (int b = 0; b < 256; ++b) m[t[(size_t)b]] = (uint8_t)b;
    return m;
  }();
  return table;
}

// ---------------------------------------------------------------------------
// Pre-tokenizer: Split(\p{N}{1,3}) then the GPT-4 style word regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// ---------------------------------------------------------------------------

namespace {

const size_t kNPos = std::string::npos;

size_t match_contraction(const std::string& s, size_t p) {
  if (p >= s.size() || s[p] != '\'') return kNPos;
  auto ci = [](char c) { return (char)std::tolower((unsigned char)c); };
  auto at = [&](size_t i) { return i < s.size() ? ci(s[i]) : '\0'; };
  const size_t q = p + 1;
  switch (at(q)) {
    case 's': return q + 1;
    case 't': return q + 1;
    case 'r': return at(q + 1) == 'e' ? q + 2 : kNPos;
    case 'v': return at(q + 1) == 'e' ? q + 2 : kNPos;
    case 'm': return q + 1;
    case 'l': return at(q + 1) == 'l' ? q + 2 : kNPos;
    case 'd': return q + 1;
    default: return kNPos;
  }
}

size_t match_letter_run(const std::string& s, size_t q) {
  size_t r = q;
  bool any = false;
  while (r < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, r, cp);
    if (len <= 0 || !cp_is_letter(cp)) break;
    r += (size_t)len;
    any = true;
  }
  return any ? r : kNPos;
}

size_t match_letters(const std::string& s, size_t p) {
  if (p < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, p, cp);
    if (len > 0 && cp != '\r' && cp != '\n' && !cp_is_letter(cp) && !cp_is_number(cp)) {
      size_t e = match_letter_run(s, p + (size_t)len);
      if (e != kNPos) return e;
    }
  }
  return match_letter_run(s, p);
}

size_t match_numbers(const std::string& s, size_t p) {
  size_t r = p;
  bool any = false;
  while (r < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, r, cp);
    if (len <= 0 || !cp_is_number(cp)) break;
    r += (size_t)len;
    any = true;
  }
  return any ? r : kNPos;
}

size_t match_punct(const std::string& s, size_t p) {
  size_t q = p;
  if (q < s.size() && s[q] == ' ') q++;
  size_t r = q;
  bool any = false;
  while (r < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, r, cp);
    if (len <= 0) break;
    if (cp_is_space(cp) || cp_is_letter(cp) || cp_is_number(cp)) break;
    r += (size_t)len;
    any = true;
  }
  if (!any) return kNPos;
  while (r < s.size() && (s[r] == '\r' || s[r] == '\n')) r++;
  return r;
}

size_t match_space_crlf(const std::string& s, size_t p) {
  size_t q = p;
  size_t last_crlf = kNPos;
  while (q < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, q, cp);
    if (len <= 0 || !cp_is_space(cp)) break;
    if (cp == '\r' || cp == '\n') last_crlf = q;
    q += (size_t)len;
  }
  if (last_crlf == kNPos) return kNPos;
  size_t e = last_crlf;
  while (e < s.size() && (s[e] == '\r' || s[e] == '\n')) e++;
  return e;
}

size_t match_space_no_nonspace(const std::string& s, size_t p) {
  size_t q = p;
  while (q < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, q, cp);
    if (len <= 0 || !cp_is_space(cp)) break;
    q += (size_t)len;
  }
  if (q == p) return kNPos;
  if (q == s.size()) return q; // (?!\S) holds at end of input
  // Step back one character so the lookahead sees whitespace.
  size_t back = p;
  size_t cur = p;
  while (cur < q) { back = cur; uint32_t cp; cur += (size_t)utf8_decode(s, cur, cp); }
  if (back == p) return kNPos;
  return back;
}

size_t match_space(const std::string& s, size_t p) {
  size_t q = p;
  while (q < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, q, cp);
    if (len <= 0 || !cp_is_space(cp)) break;
    q += (size_t)len;
  }
  return q > p ? q : kNPos;
}

void split_digits(const std::string& s, std::vector<std::pair<size_t, size_t>>& out) {
  size_t p = 0, run_start = 0;
  const size_t n = s.size();
  while (p < n) {
    uint32_t cp;
    int len = utf8_decode(s, p, cp);
    if (len <= 0) { p++; continue; }
    if (!cp_is_number(cp)) { p += (size_t)len; continue; }
    size_t q = p;
    int count = 0;
    while (q < n && count < 3) {
      uint32_t c2;
      int l2 = utf8_decode(s, q, c2);
      if (l2 <= 0 || !cp_is_number(c2)) break;
      q += (size_t)l2;
      count++;
    }
    if (p > run_start) out.emplace_back(run_start, p);
    out.emplace_back(p, q);
    p = q;
    run_start = p;
  }
  if (run_start < n) out.emplace_back(run_start, n);
}

void split_word_regex(const std::string& s, std::vector<std::pair<size_t, size_t>>& out) {
  const size_t n = s.size();
  size_t p = 0;
  while (p < n) {
    size_t e = match_contraction(s, p);
    if (e == kNPos) e = match_letters(s, p);
    if (e == kNPos) e = match_numbers(s, p);
    if (e == kNPos) e = match_punct(s, p);
    if (e == kNPos) e = match_space_crlf(s, p);
    if (e == kNPos) e = match_space_no_nonspace(s, p);
    if (e == kNPos) e = match_space(s, p);
    if (e == kNPos) { // uncovered character: emit alone
      uint32_t cp;
      int len = utf8_decode(s, p, cp);
      if (len <= 0) len = 1;
      e = p + (size_t)len;
    }
    out.emplace_back(p, e);
    p = e;
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool Tokenizer::load(const std::string& dir, std::string& err) {
  json j;
  if (!parse_json_file(dir + "/tokenizer.json", j, err)) return false;

  try {
    const auto& model = j.at("model");
    if (model.at("type").get<std::string>() != "BPE") {
      err = "tokenizer.json: only BPE models are supported";
      return false;
    }
    int64_t max_id = -1;
    for (auto& [tok, id] : model.at("vocab").items()) {
      int32_t v = id.get<int32_t>();
      vocab_[tok] = v;
      max_id = std::max<int64_t>(max_id, v);
    }
    for (auto& a : j.at("added_tokens")) {
      max_id = std::max<int64_t>(max_id, a.at("id").get<int32_t>());
    }
    id_text_.assign((size_t)max_id + 1, std::string());
    flags_.assign((size_t)max_id + 1, 0);
    for (auto& [tok, id] : model.at("vocab").items()) {
      int32_t v = id.get<int32_t>();
      if (id_text_[(size_t)v].empty()) id_text_[(size_t)v] = tok;
    }
    for (auto& a : j.at("added_tokens")) {
      int32_t id = a.at("id").get<int32_t>();
      std::string content = a.at("content").get<std::string>();
      id_text_[(size_t)id] = content;
      flags_[(size_t)id] |= 1;
      if (a.value("special", false)) flags_[(size_t)id] |= 2;
      AddedTok t;
      t.content = content;
      t.id = id;
      t.single_word = a.value("single_word", false);
      added_.push_back(std::move(t));
    }
    std::sort(added_.begin(), added_.end(), [](const AddedTok& a, const AddedTok& b) {
      return a.content.size() > b.content.size();
    });
    for (size_t i = 0; i < added_.size(); ++i) {
      if (!added_[i].content.empty())
        added_by_first_[(uint8_t)added_[i].content[0]].push_back((int)i);
    }
    int rank = 0;
    for (auto& merge : model.at("merges")) {
      std::string a = merge.at(0).get<std::string>();
      std::string b = merge.at(1).get<std::string>();
      merges_.emplace(a + "\x01" + b, rank++);
    }
    max_merge_rank_ = rank;
  } catch (const std::exception& e) {
    err = std::string("tokenizer.json: ") + e.what();
    return false;
  }

  // Every byte must exist as a single-character token for byte-level BPE.
  const auto& b2c = byte_to_cp_table();
  for (int b = 0; b < 256; ++b) {
    std::string s;
    utf8_append(s, b2c[(size_t)b]);
    if (!vocab_.count(s)) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "tokenizer.json: byte %d has no single-token entry", b);
      err = buf;
      return false;
    }
  }
  return true;
}

int Tokenizer::id_of(const std::string& token) const {
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::text_of(int id) const {
  if (id < 0 || id >= (int)id_text_.size()) return std::string();
  return id_text_[(size_t)id];
}

bool Tokenizer::is_special(int id) const {
  if (id < 0 || id >= (int)flags_.size()) return false;
  return (flags_[(size_t)id] & 2) != 0;
}

bool Tokenizer::is_added(int id) const {
  if (id < 0 || id >= (int)flags_.size()) return false;
  return (flags_[(size_t)id] & 1) != 0;
}

// ---------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------

static std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp;
    int len = utf8_decode(s, i, cp);
    if (len <= 0) len = 1;
    out.push_back(s.substr(i, (size_t)len));
    i += (size_t)len;
  }
  return out;
}

void Tokenizer::bpe(const std::string& segment, std::vector<int>& out) const {
  std::vector<std::string> syms = utf8_chars(segment);
  // Repeatedly merge the lowest-rank adjacent pair (leftmost on ties).
  while (syms.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = syms.size();
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      auto it = merges_.find(syms[i] + "\x01" + syms[i + 1]);
      if (it != merges_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i == syms.size()) break;
    syms[best_i] += syms[best_i + 1];
    syms.erase(syms.begin() + (long)best_i + 1);
  }
  for (auto& s : syms) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      continue;
    }
    // Defensive: fall back to the individual byte tokens.
    for (auto& c : utf8_chars(s)) {
      auto i2 = vocab_.find(c);
      if (i2 != vocab_.end()) out.push_back(i2->second);
    }
  }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
  std::vector<int> ids;
  if (add_bos) {
    auto it = vocab_.find("<s>");
    if (it != vocab_.end()) ids.push_back(it->second);
  }
  if (text.empty()) return ids;

  const auto& b2c = byte_to_cp_table();
  std::vector<std::pair<size_t, size_t>> segs, words;

  auto emit_plain = [&](const std::string& chunk) {
    segs.clear();
    words.clear();
    split_digits(chunk, segs);
    for (auto& [a, b] : segs) {
      size_t before = words.size();
      split_word_regex(chunk.substr(a, b - a), words);
      // remap to absolute offsets
      for (size_t i = before; i < words.size(); ++i) {
        words[i].first += a;
        words[i].second += a;
      }
    }
    for (auto& [a, b] : words) {
      std::string encoded;
      encoded.reserve(b - a);
      for (size_t i = a; i < b; ++i) utf8_append(encoded, b2c[(uint8_t)chunk[i]]);
      bpe(encoded, ids);
    }
  };

  size_t pos = 0;
  std::string plain;
  while (pos < text.size()) {
    // Longest added-token match at this position.
    int best_id = -1;
    size_t best_len = 0;
    auto bucket = added_by_first_.find((uint8_t)text[pos]);
    if (bucket != added_by_first_.end()) {
      for (int idx : bucket->second) {
        const AddedTok& t = added_[idx];
        if (t.content.empty() || text.size() - pos < t.content.size()) continue;
        if (t.content.size() <= best_len) continue;
        if (text.compare(pos, t.content.size(), t.content) != 0) continue;
        if (t.single_word) {
          bool ok = true;
          if (pos > 0) {
            uint32_t cp;
            int len = utf8_decode(text, pos - 1, cp);
            (void)len;
            if (cp_is_letter(cp) || cp_is_number(cp) || cp == '_') ok = false;
          }
          if (pos + t.content.size() < text.size()) {
            uint32_t cp;
            utf8_decode(text, pos + t.content.size(), cp);
            if (cp_is_letter(cp) || cp_is_number(cp) || cp == '_') ok = false;
          }
          if (!ok) continue;
        }
        best_id = t.id;
        best_len = t.content.size();
      }
    }
    if (best_id >= 0) {
      if (!plain.empty()) { emit_plain(plain); plain.clear(); }
      ids.push_back(best_id);
      pos += best_len;
      continue;
    }
    plain.push_back(text[pos]);
    pos++;
  }
  if (!plain.empty()) emit_plain(plain);
  return ids;
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

static void token_bytes(const std::string& text, std::string& out) {
  const auto& c2b = cp_to_byte_table();
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp;
    int len = utf8_decode(text, i, cp);
    if (len <= 0) len = 1;
    auto it = c2b.find(cp);
    if (it != c2b.end()) out.push_back((char)it->second);
    else out.append(text, i, (size_t)len); // passthrough (added tokens etc.)
    i += (size_t)len;
  }
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string bytes;
  for (int id : ids) {
    if (id < 0 || id >= (int)id_text_.size()) continue;
    if ((flags_[(size_t)id] & 2) && skip_special) continue;
    if (flags_[(size_t)id] & 1) {
      bytes += id_text_[(size_t)id];
      continue;
    }
    token_bytes(id_text_[(size_t)id], bytes);
  }
  return utf8_lossy(bytes, bytes.size());
}

std::string Tokenizer::Stream::push(int id) {
  if (id < 0 || id >= (int)tok_->id_text_.size()) return std::string();
  const std::string& text = tok_->id_text_[(size_t)id];
  const bool added = (tok_->flags_[(size_t)id] & 1) != 0;
  const bool special = (tok_->flags_[(size_t)id] & 2) != 0;
  if (added) {
    std::string out;
    if (!pending_.empty()) {
      out += utf8_lossy(pending_, pending_.size());
      pending_.clear();
    }
    if (!special) out += text;
    return out;
  }
  token_bytes(text, pending_);
  size_t complete = utf8_complete_prefix(pending_);
  std::string out = utf8_lossy(pending_, complete);
  pending_.erase(0, complete);
  return out;
}

std::string Tokenizer::Stream::flush() {
  std::string out = utf8_lossy(pending_, pending_.size());
  pending_.clear();
  return out;
}

// ---------------------------------------------------------------------------
// Chat template (model's ChatML template, tool-call XML omitted)
// ---------------------------------------------------------------------------

static std::string strip_ws(const std::string& s, bool left, bool right) {
  size_t a = 0, b = s.size();
  if (left) while (a < b && (s[a] == '\n' || s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) a++;
  if (right) while (b > a && (s[b - 1] == '\n' || s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
  return s.substr(a, b - a);
}

std::string Tokenizer::render_chat(const std::vector<ChatMessage>& messages,
                                   bool add_generation_prompt, int thinking) const {
  std::string out;
  out += "<s>";
  if (!messages.empty() && messages[0].role == "system") {
    out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n";
  }
  for (size_t i = 0; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (m.role == "user" || (m.role == "system" && i != 0)) {
      out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      std::string content = m.content;
      std::string reasoning;
      size_t close = content.find("</think>");
      if (close != std::string::npos) {
        std::string head = content.substr(0, close);
        size_t open = head.rfind("<think>");
        reasoning = strip_ws(open == std::string::npos ? head : head.substr(open + 7), true, true);
        content = strip_ws(content.substr(close + 8), true, false);
      }
      out += "<|im_start|>assistant\n";
      if (!reasoning.empty()) {
        out += "<think>\n" + reasoning + "\n</think>\n\n" + content;
      } else if (content.find("<think>") == std::string::npos &&
                 content.find("</think>") == std::string::npos) {
        out += "<think>\n\n</think>\n\n" + content;
      } else {
        out += content;
      }
      out += "<|im_end|>\n";
    } else if (m.role == "tool") {
      const bool first_of_run = (i == 0) || (messages[i - 1].role != "tool");
      const bool last_of_run = (i + 1 == messages.size()) || (messages[i + 1].role != "tool");
      if (first_of_run) out += "<|im_start|>user";
      out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
      if (last_of_run) out += "<|im_end|>\n";
    }
    // Roles the template ignores (e.g. unknown roles) are skipped.
  }
  if (add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    if (thinking == 0) out += "<think>\n\n</think>\n\n";
    else if (thinking > 0) out += "<think>\n";
  }
  return out;
}

} // namespace exl3
