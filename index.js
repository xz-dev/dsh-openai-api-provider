/**
 * dsh-openai-api-provider — an OpenAI-compatible Responses gateway as a
 * DeepSeek Harness LLM provider route with a live model catalog.
 *
 * The request path is dsh-llm-pi-ai's exported `PiAiAdapter` (stream
 * conversion, replay, images, retry facts); this plugin only supplies its
 * provider profile, rebuilt from the gateway's `GET /models`:
 *
 *  - startup registers the route from the cached catalog, then refreshes in
 *    the background (first run with no cache waits once, bounded);
 *  - `/gateway refresh` refreshes in-session; a changed catalog swaps the
 *    profile Map and calls `registration.replace`, which publishes
 *    `llm/adapters-updated` so model pickers re-read the route;
 *  - the key resolves per request through the credentials service, falling
 *    back to the launch environment — no launcher, no profile patch writes.
 *
 * @module dsh-openai-api-provider
 */
import schema from "@deepseek-ai/schemastery";
import { PiAiAdapter } from "@deepseek-ai/dsh-llm-pi-ai";
import { LlmError, RetryPolicySchema, assertUsableApiKey, resolveImageAttachmentAccess, resolveRetryPolicy } from "@deepseek-ai/dsh-llm";
import { credentialRef, isCredentialRefName } from "@deepseek-ai/dsh-credentials";
import { launchEnvironmentOf } from "@deepseek-ai/dsh-launch-environment";
import { dshHomePath } from "@deepseek-ai/dsh-home-paths";
import { createProvider } from "@earendil-works/pi-ai";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { fetchCatalog, normalizeBaseUrl, readCache, redact, toPiModels, writeCache } from "./lib/catalog.js";

export const name = "dsh-openai-api-provider";
// credentials: apply waits for the service so the startup base URL/key lookup
// does not race its async init and silently fall back to the environment.
export const inject = ["llm", "credentials"];

export const Config = schema.object({
	provider: schema.string().min(1).default("gateway").description("Provider route id models are addressed by (route/model)."),
	displayName: schema.string().min(1).default("OpenAI API Gateway"),
	baseURL: schema.string().description("Gateway base URL (…/v1). Overrides baseURLEnv."),
	baseURLEnv: schema.string().default("OPENAI_API_EXTENSION_BASE_URL").description("Credential/environment name holding the base URL."),
	apiKeyEnv: schema.string().default("OPENAI_API_EXTENSION_API_KEY").description("Credential/environment name holding the API key."),
	clientVersion: schema.string().default("0.1.0").description("Sent as client_version on GET /models; Codex-style gateways pick the served catalog by it. Empty omits it."),
	command: schema.string().min(1).default("gateway").description("Slash command name for status/refresh."),
	firstFetchTimeoutMs: schema.number().step(1).min(1).default(30000),
	cacheFile: schema.string().description("Catalog cache file; default $DSH_HOME/cache/dsh-openai-api-provider/<provider>.json."),
	retryPolicy: RetryPolicySchema,
	streamIdleTimeoutMs: schema.number().min(1).default(300000),
});

const DEFAULT_IMAGE_LIMITS = { maxRequestImageBytes: 20 * 1024 * 1024, requestImagePixelBudget: 2048 * 2048, requestImageMaxBytes: 1024 * 1024 };

/** Harness api-key auth: the per-request key PiAiAdapter passes wins in pi-ai's Models. */
function requestKeyAuth(displayName) {
	return {
		apiKey: {
			name: displayName,
			resolve: ({ credential }) => Promise.resolve({ auth: credential?.key === undefined ? {} : { apiKey: credential.key }, source: displayName }),
		},
	};
}

/**
 * Build the one-route profile Map PiAiAdapter reads. A new Map identity makes
 * the adapter build a new snapshot; in-flight calls keep the one they hold.
 */
export function buildProfiles({ provider, displayName, baseUrl, entries, retryPolicy, streamIdleTimeoutMs, apiKeyEnv }) {
	const models = baseUrl === undefined ? [] : toPiModels(entries, { provider, baseUrl });
	return new Map([
		[
			provider,
			{
				provider,
				displayName,
				apiKeyEnv,
				streamIdleTimeoutMs,
				...DEFAULT_IMAGE_LIMITS,
				retryPolicy,
				configuredMaxTokens: new Map(),
				modelErrors: new Map(),
				piProvider: createProvider({
					id: provider,
					name: displayName,
					...(baseUrl === undefined ? {} : { baseUrl }),
					auth: requestKeyAuth(displayName),
					models,
					api: openAIResponsesApi(),
				}),
			},
		],
	]);
}

/**
 * PiAiAdapter whose model operations wait for the initial catalog. The host
 * may resolve the default model before plugin startup finishes its first
 * fetch; without the gate that lookup fails with UNKNOWN_MODEL.
 */
export class GatedPiAiAdapter extends PiAiAdapter {
	ready;
	constructor(options, ready) {
		super(options);
		this.ready = ready;
	}
	async listModels(provider) {
		await this.ready;
		return super.listModels(provider);
	}
	async resolveModel(provider, model, signal) {
		await this.ready;
		return super.resolveModel(provider, model, signal);
	}
	async prepareCall(provider, model, signal) {
		await this.ready;
		return super.prepareCall(provider, model, signal);
	}
	async *stream(options) {
		await this.ready;
		yield* super.stream(options);
	}
}

/** Resolve a named value: credentials service first, then the launch environment. */
async function lookup(ctx, name) {
	if (!name) return undefined;
	if (isCredentialRefName(name)) {
		const hit = await ctx.get("credentials")?.resolve(credentialRef(name));
		if (hit?.value) return hit.value;
	}
	const value = launchEnvironmentOf(ctx).get(name)?.value;
	return value ? value : undefined;
}

export async function apply(ctx, config) {
	const cfg = Config(config ?? {});
	const { provider, displayName } = cfg;
	const retryPolicy = resolveRetryPolicy(cfg.retryPolicy, `${name}: provider "${provider}" retryPolicy`);
	const cacheFile = cfg.cacheFile ?? dshHomePath("cache", name, `${provider}.json`);
	const log = ctx.logger(name);
	const state = { baseUrl: undefined, entries: [], checkedAt: undefined, lastError: undefined, inFlight: undefined };

	const resolveApiKey = async (route) => {
		const raw = await lookup(ctx, cfg.apiKeyEnv);
		if (raw) return assertUsableApiKey(raw, name, cfg.apiKeyEnv);
		throw new LlmError(`${name}: no credential for provider route "${route}"; set ${cfg.apiKeyEnv} through the credentials service or the environment`, "MISSING_CREDENTIAL");
	};

	let profiles = buildProfiles({ provider, displayName, baseUrl: undefined, entries: [], retryPolicy, streamIdleTimeoutMs: cfg.streamIdleTimeoutMs, apiKeyEnv: cfg.apiKeyEnv });
	let markReady;
	const ready = new Promise((resolve) => (markReady = resolve));
	const adapter = new GatedPiAiAdapter({
		profiles: () => profiles,
		resolveApiKey,
		auth: {},
		resolveAttachments: () => ctx.get("attachments"),
		resolveImageAccess: (attachments, ref) => resolveImageAttachmentAccess(attachments, (hostPath) => ctx.get("fs")?.processPathFromHostPath(hostPath), ref),
		onReplayDegrade: ({ model, reason }) => log.warn(`unusable replay state on "${provider}/${model}"; sending provider-neutral content (${reason})`),
	}, ready);
	// Registered synchronously so the route exists for the fiber's lifetime;
	// catalog changes only swap the profile Map and re-announce the route.
	const registration = ctx.llm.registerAdapter([provider], adapter);

	const publish = (entries, checkedAt) => {
		state.entries = entries;
		state.checkedAt = checkedAt;
		profiles = buildProfiles({ provider, displayName, baseUrl: state.baseUrl, entries, retryPolicy, streamIdleTimeoutMs: cfg.streamIdleTimeoutMs, apiKeyEnv: cfg.apiKeyEnv });
		registration.replace([provider]);
	};

	const lifetime = new AbortController();
	ctx.effect(() => () => {
		lifetime.abort();
		markReady();
	});

	/** Fetch, map, cache and publish; one refresh at a time. Returns a summary. */
	const refresh = (timeoutMs) =>
		(state.inFlight ??= (async () => {
			let key;
			try {
				state.baseUrl ??= normalizeBaseUrl(cfg.baseURL ?? (await lookup(ctx, cfg.baseURLEnv)));
				if (state.baseUrl === undefined) throw new Error(`no base URL; set baseURL or ${cfg.baseURLEnv}`);
				key = await resolveApiKey(provider);
				const signal = timeoutMs ? AbortSignal.any([lifetime.signal, AbortSignal.timeout(timeoutMs)]) : lifetime.signal;
				const entries = await fetchCatalog(state.baseUrl, key, { signal, clientVersion: cfg.clientVersion });
				const changed = JSON.stringify(entries) !== JSON.stringify(state.entries);
				const checkedAt = Date.now();
				if (changed) publish(entries, checkedAt);
				else state.checkedAt = checkedAt;
				state.lastError = undefined;
				await writeCache(cacheFile, entries, checkedAt).catch((error) => log.warn(`catalog cache write failed: ${error.message}`));
				return { ok: true, changed, count: entries.length };
			} catch (error) {
				state.lastError = redact(error instanceof Error ? error.message : error, key);
				return { ok: false, error: state.lastError };
			} finally {
				state.inFlight = undefined;
			}
		})());

	const status = () =>
		[
			`${displayName} (${provider}):`,
			`Server: ${state.baseUrl === undefined ? "not configured" : "configured"}`,
			`Models: ${state.entries.length}`,
			`Checked: ${state.checkedAt ? new Date(state.checkedAt).toISOString() : "never"}`,
			...(state.lastError ? [`Last error: ${state.lastError}`] : []),
		].join("\n");

	ctx.inject(["commands"], (child) => {
		child.effect(() =>
			child.commands.register({
				name: cfg.command,
				description: `${displayName}: status, or refresh the model catalog`,
				input: { hint: "status | refresh" },
				handler: async (invocation) => {
					const arg = (invocation.rawInput ?? "").trim().toLowerCase();
					if (arg === "" || arg === "status") return { kind: "success", text: status() };
					if (arg !== "refresh") return { kind: "error", text: `Usage: /${cfg.command} [status|refresh]` };
					const result = await refresh(cfg.firstFetchTimeoutMs);
					if (!result.ok) return { kind: "error", text: `Refresh failed: ${result.error}` };
					return { kind: "success", text: `${result.changed ? "Updated" : "Unchanged"}: ${result.count} models.` };
				},
			}),
		);
	});

	// Startup never blocks the fiber; model operations wait on `ready` instead.
	void (async () => {
		try {
			state.baseUrl = normalizeBaseUrl(cfg.baseURL ?? (await lookup(ctx, cfg.baseURLEnv)));
			if (state.baseUrl === undefined) {
				log.warn(`no usable base URL (baseURL or ${cfg.baseURLEnv}); route "${provider}" has no models`);
				return;
			}
			const cached = await readCache(cacheFile);
			if (cached) {
				publish(cached.models, cached.checkedAt);
				void refresh().then((r) => r.ok || log.warn(`catalog refresh failed, keeping cached catalog: ${r.error}`));
				return;
			}
			const r = await refresh(cfg.firstFetchTimeoutMs);
			if (!r.ok) log.warn(`catalog fetch failed and no cache exists; route "${provider}" has no models: ${r.error}`);
		} finally {
			markReady();
		}
	})();
}
