// GLM-5 chat template rendering, checked against the checkpoint's
// chat_template.jinja (see docs/glm5.md). Expected strings are derived by
// reading that template, including its whitespace control.
// Build (repo root): clang++ -std=c++20 -O2 -Ivendor -Isrc -o /tmp/chat_test \
//   tests/chat_template_test.cpp src/tokenizer.cpp -lmlx
#include <cstdio>
#include <string>
#include <vector>
#include "tokenizer.h"
using exl3::ChatMessage;
static int fails = 0;
static void expect(const char* what, const std::string& got, const std::string& want) {
  if (got == want) { std::printf("  ok   %s\n", what); return; }
  fails++;
  std::printf("  FAIL %s\n    got : %s\n    want: %s\n", what, got.c_str(), want.c_str());
}
int main() {
  exl3::Tokenizer t; std::string err;
  if (!t.load("/tmp/glmtok", err)) { std::printf("load failed: %s\n", err.c_str()); return 1; }
  std::vector<ChatMessage> ms = {{"user", "Hello"}, {"assistant", "Hi there"}};
  // expectations below are the verbatim output of the checkpoint's
  // chat_template.jinja rendered with jinja2 (see tools/render_chat_reference.py)
  expect("plain turn",
         t.render_chat(ms, true, -1),
         "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello"
         "<|assistant|><think></think>Hi there"
         "<|assistant|><think>");
  expect("low effort, no generation prompt",
         t.render_chat(ms, false, -1, "low"),
         "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Hello"
         "<|assistant|><think></think>Hi there");
  std::vector<ChatMessage> mr = {{"user", "q"},
                                 {"assistant", "<think>\nstep 1\n</think>\nanswer"}};
  expect("reasoning content",
         t.render_chat(mr, true, -1),
         "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q"
         "<|assistant|><think>\nstep 1\n</think>answer"
         "<|assistant|><think>");
  std::vector<ChatMessage> mt = {{"user", "q"},
                                 {"assistant", "call"},
                                 {"tool", "{\"a\":1}"},
                                 {"tool", "{\"b\":2}"}};
  expect("tool observations",
         t.render_chat(mt, true, -1),
         "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>q"
         "<|assistant|><think></think>call"
         "<|observation|><tool_response>{\"a\":1}</tool_response>"
         "<tool_response>{\"b\":2}</tool_response>"
         "<|assistant|><think>");
  // clear_thinking (thinking == 0) drops prior turns' reasoning but keeps it
  // after the last user message
  std::vector<ChatMessage> mc = {{"user", "a"},
                                 {"assistant", "<think>r1</think>x"},
                                 {"user", "b"},
                                 {"assistant", "<think>r2</think>y"}};
  expect("clear_thinking=true keeps only the last turn",
         t.render_chat(mc, true, 0),
         "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>a"
         "<|assistant|><think></think>x"
         "<|user|>b"
         "<|assistant|><think>r2</think>y"
         "<|assistant|><think>");
  std::printf("chat template: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
