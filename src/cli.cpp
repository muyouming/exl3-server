#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

#include "util.h"

namespace exl3 {

namespace {

void print_stats(const GenerateResult& r) {
  const double tps = r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0;
  std::fprintf(stderr, "\n[%d prompt + %d generated tokens | prefill %.2fs | %.1f tok/s | %s]\n",
               r.prompt_tokens, r.completion_tokens, r.prefill_seconds, tps,
               r.finish_reason.c_str());
}

} // namespace

int run_cli(Engine& engine, const CliOptions& options, std::atomic<bool>& interrupted) {
  CliOptions opt = options;
  const Tokenizer& tok = engine.tokenizer();

  auto stream_out = [&](const std::string& content, const std::string& reasoning) {
    if (opt.show_reasoning && !reasoning.empty()) {
      std::fputs("\033[2m", stdout);
      std::fwrite(reasoning.data(), 1, reasoning.size(), stdout);
      std::fputs("\033[0m", stdout);
    }
    if (!content.empty()) std::fwrite(content.data(), 1, content.size(), stdout);
    std::fflush(stdout);
  };

  auto generate_ids = [&](const std::vector<int>& ids) {
    interrupted.store(false);
    opt.gen.start_in_think = (opt.thinking > 0);
    opt.gen.split_think = !opt.raw; // raw completions keep <think> markers inline
    GenerateResult r = engine.generate(ids, opt.gen, stream_out, [&] {
      return interrupted.load();
    });
    std::fflush(stdout);
    if (opt.stats) print_stats(r);
    return r;
  };

  // One-shot prompts.
  for (auto& p : opt.prompts) {
    std::vector<int> ids;
    if (opt.raw) {
      ids = tok.encode(p, /*add_bos=*/true);
    } else {
      std::vector<ChatMessage> msgs;
      if (!opt.system.empty()) msgs.push_back({"system", opt.system});
      msgs.push_back({"user", p});
      ids = tok.encode(tok.render_chat(msgs, /*add_generation_prompt=*/true, opt.thinking),
                       /*add_bos=*/false);
    }
    generate_ids(ids);
    std::fputc('\n', stdout);
  }
  if (!opt.interactive) return 0;

  // Interactive REPL.
  std::vector<ChatMessage> history;
  std::string line;
  const bool tty = ::isatty(0);
  for (;;) {
    if (tty) {
      std::fputs("\033[1m> \033[0m", stdout);
      std::fflush(stdout);
    }
    if (!std::getline(std::cin, line)) break;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset" || line == "/clear") {
      history.clear();
      engine.reset();
      std::fputs("(history cleared)\n", stdout);
      continue;
    }
    if (line == "/help") {
      std::fputs("commands: /exit /reset /think /nothink /system <text> /raw <text>\n", stdout);
      continue;
    }
    if (line == "/think") { opt.thinking = 1; std::fputs("(thinking enabled)\n", stdout); continue; }
    if (line == "/nothink") { opt.thinking = 0; std::fputs("(thinking disabled)\n", stdout); continue; }
    if (line.rfind("/system ", 0) == 0) {
      opt.system = line.substr(8);
      history.clear();
      engine.reset();
      std::fputs("(system prompt set, history cleared)\n", stdout);
      continue;
    }
    if (line.rfind("/raw ", 0) == 0) {
      generate_ids(tok.encode(line.substr(5), /*add_bos=*/true));
      std::fputc('\n', stdout);
      continue;
    }
    if (line.empty()) continue;

    std::vector<ChatMessage> msgs;
    if (!opt.system.empty()) msgs.push_back({"system", opt.system});
    for (auto& m : history) msgs.push_back(m);
    msgs.push_back({"user", line});

    std::vector<int> ids = tok.encode(tok.render_chat(msgs, /*add_generation_prompt=*/true, opt.thinking),
                                      /*add_bos=*/false);
    GenerateResult r = generate_ids(ids);
    std::fputc('\n', stdout);
    history.push_back({"user", line});
    history.push_back({"assistant", r.text});
  }
  return 0;
}

} // namespace exl3
