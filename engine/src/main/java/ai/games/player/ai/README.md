# Model-backed players

Run commands below from `engine/`. See [engine setup](../../../../../../../README.md), [game rules](../../../../../../../RULES.md), and [result-suite instructions](../../../../../../test/java/ai/games/results/README.md). Model-backed result runs are opt-in and are excluded from the current rules refresh.

Ollama model selection:
- Model defaults are declared in the player classes; select the model explicitly with `-Dollama.model=<model>` for reproducible runs.
- Override per run: `./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama" "-Dollama.model=mistral-large:123b"`
- Or set env: `$env:OLLAMA_MODEL="mistral-large:123b"; ./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-ollama"`
- Each `OllamaPlayer` instance represents one game. It establishes and pins the selected profile's strategy before the deal, then retains a rolling window of recent board turns.
- Choose the shared game prompt profile with `-Dgame.prompt.profile=p0|detailed`. `p0` (the default, version P0) asks the model to supply its existing knowledge. `detailed` (version P1.5) supplies corrected engine rules, strategic priorities, progress requirements, and a more explicit game interface. A player captures one matched policy for the whole game. The older `-Dllm.prompt.profile` property remains accepted for reproducibility.
- Configure the rolling window with `-Dollama.memory.turns=8` and the Ollama context budget with `-Dollama.context.tokens=32768`. Input truncation is disabled so an unexpected overflow fails visibly.
- Thinking defaults to `off` so thinking-capable local models cannot spend an unbounded number of tokens choosing one move. Use `-Dollama.thinking=auto|on|low|medium|high` for controlled reasoning experiments and cap each response with `-Dollama.max.output.tokens=1024`.

OpenAI setup:
- Add an OpenAI-compatible API key via either:
  - Spring property: `-Dopenai.apiKey=sk-...` (or in `application.properties`), or
  - Environment variable: `OPENAI_API_KEY=sk-...`.
- Select the chat model with `-Dopenai.model=<model>`.
- Run the OpenAI-backed player with:
  `./gradlew bootRun --console=plain "-Dspring.profiles.active=ai-openai"`

Codex CLI setup (ChatGPT subscription rather than API billing):
- Install Codex CLI and sign in with ChatGPT; `codex login status` must report `Logged in using ChatGPT`.
- Configure the model with `-Dcodex.model` and reasoning effort with `-Dcodex.reasoning.effort`.
- Use `-Dcodex.models=gpt-5.6-luna,gpt-5.6-sol,gpt-6-astra` to compare several models in one benchmark.
- Each game uses a new persistent Codex session. Before the first board, the selected prompt profile establishes the strategy; every move then resumes that session so the model retains its strategy, earlier boards, commands, and feedback. Sessions are never reused across games.
- Codex result runs disable engine guidance. With the default P0 profile, the benchmark measures the model's self-authored strategy rather than engine recommendations.
- Transient capacity, timeout, and transport failures are retried up to five times with exponential backoff. Override this with `-Dcodex.retry.max.attempts` and `-Dcodex.retry.initial.delay.millis`; authentication and usage-limit failures are not retried.
- The player refuses non-ChatGPT authentication and removes API-key environment variables from CLI subprocesses.
- Run a small benchmark first:
  `./gradlew test --tests ai.games.results.CodexCliPlayerResultsTest --console=plain --rerun-tasks "-Dcodex.tests=true" "-Dcodex.models=gpt-5.6-luna,gpt-5.6-sol,gpt-6-astra" "-Dtest.games=1"`

GitHub Copilot CLI setup:
- Install `copilot`, authenticate it with the GitHub account that owns the Copilot subscription, and use `/model` interactively to confirm model availability.
- Configure one model with `-Dcopilot.model` or a sweep with `-Dcopilot.models`; the default is `claude-haiku-4.5`.
- Each game uses a fresh persistent Copilot session and the same selected prompt profile as the Codex player. Copilot runs in an empty temporary directory with custom instructions, MCP servers, and tools disabled.
- Copilot has no strict response-schema flag, so the player validates returned JSON against the engine's legal moves and makes up to three in-session formatting corrections. Configure this with `-Dcopilot.response.max.attempts`.
- Copilot result runs disable engine guidance and log the session ID plus Copilot's reported premium-request usage for every completed game.
- Set `SOLITAIRE_LOG_DIR` when running model sweeps concurrently so each process owns its rolling `game.log` and `episode.log`. For example, use `SOLITAIRE_LOG_DIR=logs/sonnet` for a Sonnet run while another process uses the default `logs` directory.
- Run a one-game smoke test with a 200-move cap:
  `./gradlew test --tests ai.games.results.CopilotCliPlayerResultsTest --console=plain --rerun-tasks "-Dcopilot.tests=true" "-Dcopilot.models=claude-haiku-4.5" "-Dtest.games=1" "-Dtest.max.moves.per.game=200" "-Dlog.episodes=true"`

TypeSafe AI setup:
- Set `TYPESAFE_API_KEY` in the environment or pass `-Dtypesafe.apiKey`. The repository's ignored `experiments/runtime/typesafe.env` file can be sourced before a run.
- The player pins `jev-1.13.0` by default for reproducibility. Override it with `-Dtypesafe.model` and override the endpoint with `-Dtypesafe.baseUrl` when needed.
- Each turn sends the current human-visible structured board plus compact observations, commands, and feedback from every previous turn to a System One Choice question. Historical observations retain foundation/tableau tops, hidden counts, and stock/waste state so card memory survives without exceeding Jev's context window. Jev returns an opaque option that maps to one engine-generated legal command.
- TypeSafe uses the shared `-Dgame.prompt.profile=p0|detailed` switch. P0 supplies no strategic policy; detailed/P1.5 adds corrected rules, eight decision priorities, and progress requirements shared with the conversational model players. Engine guidance remains disabled for both.
- TypeSafe supplies the detailed policy with one-step facts by simulating each legal command. It reports hidden-card and foundation progress, repeated-position signals, and stalls, and marks the strongest immediate-progress options without removing any legal choices. Episode metadata records the shared `policy_version`.
- Episode steps include Jev's model version, confidence, option probabilities, latency, and token usage when `-Dlog.episodes=true` is enabled.
- Transient rate limits and server or transport failures retry up to five times. Configure this with `-Dtypesafe.retry.max.attempts` and `-Dtypesafe.retry.initial.delay.millis`.
- Run a one-game smoke test with a 200-move cap:
  `source ../experiments/runtime/typesafe.env && SOLITAIRE_LOG_DIR=logs/typesafe ./gradlew test --tests ai.games.results.TypeSafePlayerResultsTest --console=plain --rerun-tasks "-Dtypesafe.tests=true" "-Dtest.games=1" "-Dtest.max.moves.per.game=200" "-Dlog.episodes=true"`
