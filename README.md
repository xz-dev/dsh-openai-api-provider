# dsh-openai-api-provider

An OpenAI-compatible Responses gateway as a [DeepSeek Harness](https://github.com/deepseek-ai) LLM provider route, with a live `GET /models` catalog and in-session refresh.

It reuses `@deepseek-ai/dsh-llm-pi-ai`'s exported `PiAiAdapter` for the request path (stream conversion, replay, images, retry facts) and only supplies the provider profile, rebuilt from the gateway catalog. No launcher script, no `--patch` overlay, no writes to profile config.

## Behavior

- **Startup:** the route is registered immediately. The cached catalog (`$DSH_HOME/cache/dsh-openai-api-provider/<provider>.json`, mode 0600, no secrets) is published at once and refreshed in the background. With no cache, the first fetch runs in the background; model operations wait for it (bounded by `firstFetchTimeoutMs`) so the default model resolves.
- **`/gateway`** (or `/gateway status`) shows configured/not, model count, last check, last error. **`/gateway refresh`** refetches; a changed catalog swaps the profile and calls `registration.replace`, which publishes `llm/adapters-updated` so model pickers re-read the route. An unchanged catalog does not re-announce.
- **Credentials:** `apiKeyEnv` / `baseURLEnv` resolve through the `credentials` service first, then the launch environment. The key is sent only as the bearer header and is redacted from every error.
- **Mapping** (same rules as `openai-api-pi-extension`): only conversational text models; reasoning levels from `supported_reasoning_levels` or `capabilities.effort_tiers` (`minimal` aliases `low`; `none`-only means non-reasoning); missing context window → 128K; an output cap not below the context window is ignored; `client_version` is sent because Codex-style gateways select the served catalog by it.

## Config

| key | default | |
|---|---|---|
| `provider` | `gateway` | route id (`route/model`) |
| `displayName` | `OpenAI API Gateway` | |
| `baseURL` | — | overrides `baseURLEnv` |
| `baseURLEnv` | `OPENAI_API_EXTENSION_BASE_URL` | |
| `apiKeyEnv` | `OPENAI_API_EXTENSION_API_KEY` | |
| `clientVersion` | `0.1.0` | `client_version` query on `/models`; empty omits |
| `command` | `gateway` | slash command name |
| `firstFetchTimeoutMs` | `30000` | |
| `cacheFile` | see above | |
| `retryPolicy` | dsh-llm default | same schema as llm-pi-ai routes |
| `streamIdleTimeoutMs` | `300000` | |

The bundle patch inserts one instance with the defaults above and `retryPolicy: {mode: normal, maxRetries: 10}`. Do not also declare a `gateway` route in `llm-pi-ai`; routes are exclusive.

## Known gaps vs. the Pi extension

- No `/login` flow; store the key through the credentials service or the environment.
- No Responses WebSocket bridge (`transport: websocket`); requests use SSE.
- No empty-message filtering before dispatch.
- No periodic background refresh; refresh happens at startup and on `/gateway refresh`.
- `ResolvedPiAiProviderProfile` is not an entry export of dsh-llm-pi-ai; peers are pinned to `0.1.7-rc.2` / pi-ai `0.85.1`. Re-run the tests before upgrading DSH.

## Test

```
pnpm install --ignore-scripts
node --test test/*.test.mjs
```

Tests use a local HTTP stub; no network or credentials.

## License

MIT
