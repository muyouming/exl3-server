#pragma once
// HuggingFace `tokenizer.json` (BPE + ByteLevel) and the model's ChatML chat
// template. Supports added tokens, the model's two-stage pre-tokenizer
// (\\p{N}{1,3} split, then the GPT-4 style word regex), streaming UTF-8
// detokenization, and chat rendering with an optional thinking block.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace exl3 {

struct ChatMessage {
  std::string role;
  std::string content;
};

class Tokenizer {
 public:
  bool load(const std::string& dir, std::string& err);

  // Encode text. Added/special tokens present in the text are emitted as single
  // ids; everything else goes through the pre-tokenizer and BPE.
  std::vector<int> encode(const std::string& text, bool add_bos = false) const;

  // Pre-tokenizer output only (diagnostics/tests): the pieces BPE then works
  // on, which must not cross any piece boundary.
  std::vector<std::string> pretokenize(const std::string& text) const;

  // Full decode (invalid UTF-8 becomes U+FFFD). Special tokens are skipped when
  // skip_special is set.
  std::string decode(const std::vector<int>& ids, bool skip_special = true) const;

  // Token-by-token decode that only emits complete UTF-8 sequences.
  class Stream {
   public:
    Stream(const Tokenizer* tok) : tok_(tok) {}
    std::string push(int id);
    std::string flush();

   private:
    const Tokenizer* tok_;
    std::string pending_;
  };

  // Render the model's chat template. `thinking` < 0 leaves the template's
  // enable_thinking undefined, 0 renders an empty think block, 1 opens one.
  std::string render_chat(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt, int thinking = 0) const;

  int vocab_size() const { return (int)id_text_.size(); }
  int id_of(const std::string& token) const;
  std::string text_of(int id) const;
  bool is_special(int id) const;
  bool is_added(int id) const;

 private:
  void bpe(const std::string& segment, std::vector<int>& out) const;

  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<std::string, int32_t> merges_; // "a\x01b" -> rank
  std::vector<std::string> id_text_;
  std::vector<uint8_t> flags_; // 1 = added token, 2 = special
  int max_merge_rank_ = 0;

  struct AddedTok {
    std::string content;
    int32_t id;
    bool single_word;
  };
  std::vector<AddedTok> added_;
  std::unordered_map<uint8_t, std::vector<int>> added_by_first_;
};

} // namespace exl3
