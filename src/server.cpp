#include "server.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "util.h"

using json = nlohmann::json;

namespace exl3 {

namespace {

int64_t epoch_seconds() {
  return (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string make_id(const char* prefix) {
  static std::atomic<uint64_t> counter{0};
  uint64_t n = counter.fetch_add(1) + 1;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%s-%llx-%llx", prefix, (unsigned long long)now_ms(),
                (unsigned long long)n);
  return buf;
}

void api_error(httplib::Response& res, int status, const std::string& message,
               const char* type = "invalid_request_error") {
  json e;
  e["error"] = {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}};
  res.status = status;
  res.set_content(e.dump(), "application/json");
}

bool parse_body(const httplib::Request& req, httplib::Response& res, json& out,
                std::string& err) {
  try {
    out = json::parse(req.body);
  } catch (const std::exception& e) {
    err = std::string("invalid JSON body: ") + e.what();
    return false;
  }
  if (!out.is_object()) {
    err = "request body must be a JSON object";
    return false;
  }
  return true;
}

std::string content_to_text(const json& c) {
  if (c.is_string()) return c.get<std::string>();
  if (c.is_array()) {
    std::string out;
    for (auto& part : c) {
      if (part.is_string()) {
        out += part.get<std::string>();
      } else if (part.is_object()) {
        if (part.contains("text") && part["text"].is_string()) out += part["text"].get<std::string>();
        else if (part.contains("content") && part["content"].is_string())
          out += part["content"].get<std::string>();
      }
    }
    return out;
  }
  if (c.is_null()) return std::string();
  return c.dump();
}

} // namespace

// ---------------------------------------------------------------------------
// Request parameters
// ---------------------------------------------------------------------------

struct RequestParams {
  GenerateOptions gen;
  int thinking = 0; // 0 renders an empty think block, 1 opens one
  bool stream = false;
  bool verbose = false;
  int n = 1;
  std::string model;
};

static bool parse_common_params(const json& body, RequestParams& p, httplib::Response& res) {
  auto as_int = [&](const char* key, int& dst) {
    if (body.contains(key) && body[key].is_number()) dst = (int)body[key].get<double>();
  };
  auto as_float = [&](const char* key, float& dst) {
    if (body.contains(key) && body[key].is_number()) dst = (float)body[key].get<double>();
  };

  p.stream = body.value("stream", false);
  p.model = body.value("model", std::string());
  p.gen.sample.temperature = 0.0f; // greedy unless the request asks otherwise
  as_float("temperature", p.gen.sample.temperature);
  p.gen.sample.top_p = 1.0f;
  as_float("top_p", p.gen.sample.top_p);
  p.gen.sample.min_p = 0.0f;
  as_float("min_p", p.gen.sample.min_p);
  p.gen.sample.top_k = 0;
  as_int("top_k", p.gen.sample.top_k);
  p.gen.sample.repetition_penalty = 1.0f;
  as_float("repetition_penalty", p.gen.sample.repetition_penalty);
  p.gen.sample.presence_penalty = 0.0f;
  as_float("presence_penalty", p.gen.sample.presence_penalty);
  p.gen.sample.frequency_penalty = 0.0f;
  as_float("frequency_penalty", p.gen.sample.frequency_penalty);
  if (body.contains("seed")) {
    if (body["seed"].is_number()) p.gen.sample.seed = (uint64_t)body["seed"].get<int64_t>();
    else if (body["seed"].is_null()) p.gen.sample.seed = 0;
  } else {
    p.gen.sample.seed = 0; // random
  }

  p.gen.max_tokens = 512;
  as_int("max_tokens", p.gen.max_tokens);
  if (body.contains("max_completion_tokens")) as_int("max_completion_tokens", p.gen.max_tokens);
  if (p.gen.max_tokens <= 0) p.gen.max_tokens = 1;
  p.gen.ignore_eos = body.value("ignore_eos", false);

  if (body.contains("n")) {
    if (!body["n"].is_number()) return (api_error(res, 400, "n must be a number"), false);
    p.n = (int)body["n"].get<double>();
    if (p.n < 1) p.n = 1;
    if (p.n > 8) return (api_error(res, 400, "n must be between 1 and 8"), false);
    if (p.stream && p.n > 1)
      return (api_error(res, 400, "streaming with n > 1 is not supported"), false);
  }

  if (body.contains("stop")) {
    const json& s = body["stop"];
    if (s.is_string()) {
      if (!s.get<std::string>().empty()) p.gen.stop.push_back(s.get<std::string>());
    } else if (s.is_array()) {
      for (auto& e : s) {
        if (e.is_string()) p.gen.stop.push_back(e.get<std::string>());
      }
    } else if (!s.is_null()) {
      return (api_error(res, 400, "stop must be a string or an array of strings"), false);
    }
  }

  // Thinking mode: chat_template_kwargs.enable_thinking or top-level enable_thinking.
  auto set_thinking = [&](const json& v) {
    if (v.is_boolean()) p.thinking = v.get<bool>() ? 1 : 0;
  };
  if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
    const json& kw = body["chat_template_kwargs"];
    if (kw.contains("enable_thinking")) set_thinking(kw["enable_thinking"]);
  }
  if (body.contains("enable_thinking")) set_thinking(body["enable_thinking"]);
  return true;
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

struct SseWriter {
  httplib::DataSink* sink;
  bool dead = false;
  void send(const json& j) {
    if (dead) return;
    std::string payload = "data: " + j.dump() + "\n\n";
    if (!sink->write(payload.data(), payload.size())) dead = true;
  }
  void done() {
    if (dead) return;
    const char* tail = "data: [DONE]\n\n";
    if (!sink->write(tail, std::strlen(tail))) dead = true;
  }
};

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

int run_server(Engine& engine, const ServerOptions& opt, std::atomic<bool>& stop) {
  httplib::Server svr;
  const std::string name = engine.model_name();

  svr.set_default_headers(httplib::Headers{
      {"Access-Control-Allow-Origin", "*"},
      {"Access-Control-Allow-Headers", "*"},
      {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
  });
  svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
    res.status = 204;
  });

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}, {"model", name}}.dump(), "application/json");
  });

  auto model_json = [&](const std::string& id) {
    return json{{"id", id},
                {"object", "model"},
                {"created", epoch_seconds()},
                {"owned_by", "exl3-server"}};
  };

  svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json out{{"object", "list"}, {"data", json::array({model_json(name)})}};
    res.set_content(out.dump(), "application/json");
  });
  svr.Get(R"(/v1/models/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
    res.set_content(model_json(req.matches[1]).dump(), "application/json");
  });

  // ---- /v1/chat/completions ------------------------------------------------
  svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    std::string err;
    if (!parse_body(req, res, body, err)) return api_error(res, 400, err);

    RequestParams p;
    if (!parse_common_params(body, p, res)) return;
    p.verbose = opt.verbose;
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
      return api_error(res, 400, "messages is required and must be a non-empty array");

    std::vector<ChatMessage> messages;
    for (auto& m : body["messages"]) {
      if (!m.is_object() || !m.contains("role"))
        return api_error(res, 400, "each message needs a role");
      ChatMessage cm;
      cm.role = m.value("role", std::string());
      cm.content = m.contains("content") ? content_to_text(m["content"]) : std::string();
      messages.push_back(std::move(cm));
    }
    // Tool call/tool response content is passed through as text; tool_calls XML
    // rendering from the model template is not implemented.
    p.gen.start_in_think = p.thinking > 0; // the template opens the block for us
    const std::string prompt_text = engine.tokenizer().render_chat(messages, true, p.thinking);
    std::vector<int> prompt = engine.tokenizer().encode(prompt_text, /*add_bos=*/false);
    if (prompt.empty()) return api_error(res, 400, "prompt is empty after tokenization");

    const std::string id = make_id("chatcmpl");
    const int64_t created = epoch_seconds();
    const std::string used_model = p.model.empty() ? name : p.model;

    if (p.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      res.set_chunked_content_provider(
          "text/event-stream",
          [&engine, &stop, p, prompt, id, created, used_model](size_t, httplib::DataSink& sink) -> bool {
            SseWriter sse{&sink};
            json base{{"id", id},
                      {"object", "chat.completion.chunk"},
                      {"created", created},
                      {"model", used_model}};
            {
              json first = base;
              first["choices"] = json::array({json{{"index", 0},
                                                  {"delta", json{{"role", "assistant"}, {"content", ""}}},
                                                  {"finish_reason", nullptr}}});
              sse.send(first);
            }
            std::string finish = "stop";
            auto on_delta = [&](const std::string& content, const std::string& reasoning) {
              if (sse.dead) return;
              json chunk = base;
              json delta = json::object();
              if (!content.empty()) delta["content"] = content;
              if (!reasoning.empty()) delta["reasoning_content"] = reasoning;
              if (delta.empty()) return;
              chunk["choices"] = json::array({json{{"index", 0},
                                                   {"delta", delta},
                                                   {"finish_reason", nullptr}}});
              sse.send(chunk);
            };
            GenerateResult r = engine.submit_batch(prompt, p.gen, on_delta,
                                               [&] { return sse.dead || stop.load(); });
            finish = r.finish_reason == "cancelled" ? "stop" : r.finish_reason;
            if (p.verbose) {
              std::fprintf(stderr, "[chat] %d prompt + %d gen tokens, %.1f tok/s\n",
                           r.prompt_tokens, r.completion_tokens,
                           r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0);
            }
            json last = base;
            last["choices"] = json::array({json{{"index", 0}, {"delta", json::object()},
                                                {"finish_reason", finish}}});
            if (p.verbose) {
              last["usage"] = json{{"prompt_tokens", r.prompt_tokens},
                                   {"completion_tokens", r.completion_tokens},
                                   {"total_tokens", r.prompt_tokens + r.completion_tokens}};
            }
            sse.send(last);
            sse.done();
            return false;
          });
      return;
    }

    json choices = json::array();
    int64_t prompt_tokens = 0, completion_tokens = 0;
    for (int i = 0; i < p.n; ++i) {
      GenerateResult r = engine.submit_batch(prompt, p.gen, nullptr, [&] { return stop.load(); });
      if (p.verbose)
        std::fprintf(stderr, "[chat] %d prompt + %d gen tokens, %.1f tok/s\n", r.prompt_tokens,
                     r.completion_tokens,
                     r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0);
      json message{{"role", "assistant"}, {"content", r.content}};
      if (!r.reasoning.empty()) message["reasoning_content"] = r.reasoning;
      choices.push_back(json{{"index", i},
                             {"message", message},
                             {"finish_reason", r.finish_reason == "cancelled" ? "stop" : r.finish_reason}});
      prompt_tokens = r.prompt_tokens;
      completion_tokens += r.completion_tokens;
    }
    json out{{"id", id},
             {"object", "chat.completion"},
             {"created", created},
             {"model", used_model},
             {"choices", choices},
             {"usage", json{{"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}}}};
    res.set_content(out.dump(), "application/json");
  });

  // ---- /v1/completions -----------------------------------------------------
  svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
    json body;
    std::string err;
    if (!parse_body(req, res, body, err)) return api_error(res, 400, err);

    RequestParams p;
    if (!parse_common_params(body, p, res)) return;
    p.verbose = opt.verbose;
    p.gen.split_think = false; // raw completions keep thinking markers inline
    if (!body.contains("prompt"))
      return api_error(res, 400, "prompt is required");

    std::vector<std::string> prompts;
    if (body["prompt"].is_string()) {
      prompts.push_back(body["prompt"].get<std::string>());
    } else if (body["prompt"].is_array()) {
      for (auto& e : body["prompt"]) {
        if (!e.is_string()) return api_error(res, 400, "prompt array must contain strings");
        prompts.push_back(e.get<std::string>());
      }
    } else if (body["prompt"].is_null()) {
      return api_error(res, 400, "prompt must be a string or an array of strings");
    } else {
      return api_error(res, 400, "prompt must be a string or an array of strings");
    }
    if (prompts.empty()) return api_error(res, 400, "prompt is empty");

    std::vector<std::vector<int>> encoded;
    for (auto& s : prompts) encoded.push_back(engine.tokenizer().encode(s, /*add_bos=*/true));

    const std::string id = make_id("cmpl");
    const int64_t created = epoch_seconds();
    const std::string used_model = p.model.empty() ? name : p.model;

    if (p.stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      res.set_chunked_content_provider(
          "text/event-stream",
          [&engine, &stop, p, encoded, id, created, used_model](size_t, httplib::DataSink& sink) -> bool {
            SseWriter sse{&sink};
            for (size_t idx = 0; idx < encoded.size() && !sse.dead; ++idx) {
              json base{{"id", id},
                        {"object", "text_completion"},
                        {"created", created},
                        {"model", used_model}};
              auto on_delta = [&](const std::string& content, const std::string&) {
                if (content.empty() || sse.dead) return;
                json chunk = base;
                chunk["choices"] = json::array({json{{"text", content},
                                                     {"index", (int)idx},
                                                     {"logprobs", nullptr},
                                                     {"finish_reason", nullptr}}});
                sse.send(chunk);
              };
              GenerateResult r = engine.submit_batch(encoded[idx], p.gen, on_delta,
                                                 [&] { return sse.dead || stop.load(); });
              json last = base;
              last["choices"] = json::array({json{{"text", ""},
                                                  {"index", (int)idx},
                                                  {"logprobs", nullptr},
                                                  {"finish_reason", r.finish_reason}}});
              sse.send(last);
            }
            sse.done();
            return false;
          });
      return;
    }

    json choices = json::array();
    int64_t prompt_tokens = 0, completion_tokens = 0;
    for (size_t idx = 0; idx < encoded.size(); ++idx) {
      GenerateResult r = engine.submit_batch(encoded[idx], p.gen, nullptr, [&] { return stop.load(); });
      if (p.verbose)
        std::fprintf(stderr, "[completion] %d prompt + %d gen tokens, %.1f tok/s\n",
                     r.prompt_tokens, r.completion_tokens,
                     r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0);
      choices.push_back(json{{"text", r.text},
                             {"index", (int)idx},
                             {"logprobs", nullptr},
                             {"finish_reason", r.finish_reason == "cancelled" ? "stop" : r.finish_reason}});
      prompt_tokens += r.prompt_tokens;
      completion_tokens += r.completion_tokens;
    }
    json out{{"id", id},
             {"object", "text_completion"},
             {"created", created},
             {"model", used_model},
             {"choices", choices},
             {"usage", json{{"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}}}};
    res.set_content(out.dump(), "application/json");
  });

  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                               std::exception_ptr ep) {
    std::string msg = "internal server error";
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {}
    api_error(res, 500, msg, "server_error");
  });

  std::fprintf(stderr, "exl3-server: listening on http://%s:%d (model: %s)\n", opt.host.c_str(),
               opt.port, name.c_str());
  if (!svr.listen(opt.host, opt.port)) {
    std::fprintf(stderr, "exl3-server: failed to bind %s:%d\n", opt.host.c_str(), opt.port);
    return 1;
  }
  return 0;
}

} // namespace exl3
