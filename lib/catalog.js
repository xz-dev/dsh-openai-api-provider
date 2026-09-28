/**
 * Gateway catalog: fetch `GET {baseURL}/models`, map rows to pi-ai models,
 * and persist the last good catalog. Pure except for fetch and file I/O.
 *
 * Mapping rules follow openai-api-pi-extension (and the former
 * gateway-models-patch.mjs launcher overlay):
 *  - only conversational text models (embedding/image/video/audio dropped)
 *  - reasoning levels from supported_reasoning_levels / capabilities.effort_tiers;
 *    `minimal` aliases `low`; a model offering only `none` is non-reasoning
 *  - missing context window falls back to 128K
 *  - an output cap that is not below the context window is a listing artifact
 *    and is left to the route default
 *
 * @module dsh-openai-api-provider/catalog
 */
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 32_768;
const NON_CHAT = new Set(["embedding", "image", "video", "audio"]);
const EFFORTS = new Set(["none", "low", "medium", "high", "xhigh", "max"]);
const NO_COST = Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
const CACHE_VERSION = 1;

const positive = (v) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : undefined);
const text = (v) => (typeof v === "string" && v.trim() ? v.trim() : undefined);
const segments = (s) => s.trim().toLowerCase().split(/[/:\-_]/).filter(Boolean);
const isRecord = (v) => typeof v === "object" && v !== null && !Array.isArray(v);

function identity(row) {
	const id = text(row.id) ?? text(row.slug);
	if (!id) return undefined;
	return { id, name: text(row.name) ?? text(row.display_name) ?? id };
}

function isChat(row, ident) {
	const type = typeof row.type === "string" ? row.type.trim().toLowerCase() : "";
	if (type && NON_CHAT.has(type)) return false;
	if (Array.isArray(row.output_modalities)) {
		const outs = row.output_modalities.filter((v) => typeof v === "string");
		if (outs.length && !outs.some((m) => m.trim().toLowerCase() === "text")) return false;
	}
	return ![ident.id, ident.name].some((v) => segments(v).some((s) => NON_CHAT.has(s)));
}

/**
 * pi-ai thinkingLevelMap for one row, or undefined for a non-reasoning model.
 * A level mapped to null is not offered; `off` is never offered.
 */
export function thinkingLevelMap(row) {
	const raw = row.supported_reasoning_levels ?? (isRecord(row.capabilities) ? row.capabilities.effort_tiers : undefined);
	if (!Array.isArray(raw)) return undefined;
	const set = new Set(
		raw
			.map((e) => (typeof e === "string" ? e : isRecord(e) ? e.effort : undefined))
			.filter((e) => typeof e === "string")
			.map((e) => e.trim().toLowerCase())
			.filter((e) => EFFORTS.has(e)),
	);
	if (![...set].some((e) => e !== "none")) return undefined;
	const low = set.has("low") ? "low" : null;
	const pick = (e) => (set.has(e) ? e : null);
	return { off: null, minimal: low, low, medium: pick("medium"), high: pick("high"), xhigh: pick("xhigh"), max: pick("max") };
}

/**
 * Map one catalog row to a route-independent entry, or undefined when the row
 * is not a usable chat model.
 */
export function mapRow(row) {
	if (!isRecord(row)) return undefined;
	const ident = identity(row);
	if (!ident || !isChat(row, ident)) return undefined;
	const contextWindow = positive(row.context_window) ?? positive(row.context_length) ?? DEFAULT_CONTEXT_WINDOW;
	const maxTokens = positive(row.max_output_tokens) ?? positive(row.max_completion_tokens) ?? positive(row.max_tokens);
	const inputs = Array.isArray(row.input_modalities) ? row.input_modalities : [];
	const levels = thinkingLevelMap(row);
	return {
		id: ident.id,
		name: ident.name,
		contextWindow,
		...(maxTokens !== undefined && maxTokens < contextWindow ? { maxTokens } : {}),
		input: inputs.includes("image") ? ["text", "image"] : ["text"],
		...(levels ? { thinkingLevelMap: levels } : {}),
	};
}

/** Map a raw catalog to sorted, de-duplicated entries; throws when nothing usable remains. */
export function mapCatalog(rows) {
	const seen = new Set();
	const entries = [];
	for (const row of rows) {
		const entry = mapRow(row);
		if (!entry || seen.has(entry.id)) continue;
		seen.add(entry.id);
		entries.push(entry);
	}
	if (entries.length === 0) throw new Error("model discovery returned no usable models");
	return entries.sort((a, b) => a.id.localeCompare(b.id));
}

/** Materialize entries as pi-ai `openai-responses` models for one route. */
export function toPiModels(entries, { provider, baseUrl }) {
	return entries.map((e) => ({
		id: e.id,
		name: e.name,
		api: "openai-responses",
		provider,
		baseUrl,
		input: [...e.input],
		cost: NO_COST,
		contextWindow: e.contextWindow,
		maxTokens: e.maxTokens ?? DEFAULT_MAX_TOKENS,
		reasoning: e.thinkingLevelMap !== undefined,
		...(e.thinkingLevelMap ? { thinkingLevelMap: { ...e.thinkingLevelMap } } : {}),
	}));
}

/** Replace every occurrence of a secret in a message. */
export function redact(message, secret) {
	return secret ? String(message).split(secret).join("[REDACTED]") : String(message);
}

/**
 * Fetch and map the live catalog. Thrown messages never contain the key.
 * @param {string} baseUrl - normalized base URL, no trailing slash.
 * @param {string} apiKey - bearer key.
 * @param {object} [options]
 * @param {string} [options.clientVersion] - sent as `client_version`; Codex-style
 *   gateways select the catalog they serve by it (omitted = a reduced list).
 */
export async function fetchCatalog(baseUrl, apiKey, { signal, clientVersion, fetchImpl = fetch } = {}) {
	try {
		const url = new URL(`${baseUrl}/models`);
		if (clientVersion) url.searchParams.set("client_version", clientVersion);
		const response = await fetchImpl(url, {
			headers: { Authorization: `Bearer ${apiKey}` },
			signal,
		});
		if (!response.ok) {
			const body = (await response.text().catch(() => "")).replace(/\s+/g, " ").trim().slice(0, 160);
			throw new Error(`model discovery failed: HTTP ${response.status}${body ? `: ${body}` : ""}`);
		}
		const payload = await response.json();
		const rows = Array.isArray(payload?.models) ? payload.models : Array.isArray(payload?.data) ? payload.data : undefined;
		if (!rows) throw new Error("model discovery returned an unexpected catalog shape");
		return mapCatalog(rows);
	} catch (error) {
		throw new Error(redact(error instanceof Error ? error.message : error, apiKey));
	}
}

/** Normalize a base URL: http(s) only, no credentials/query/fragment, no trailing slash. */
export function normalizeBaseUrl(raw) {
	const value = typeof raw === "string" ? raw.trim().replace(/\/+$/, "") : "";
	if (!value) return undefined;
	try {
		const url = new URL(value);
		if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password || url.search || url.hash) return undefined;
		return url.toString().replace(/\/+$/, "");
	} catch {
		return undefined;
	}
}

/** Read the cached catalog; undefined when absent or unreadable. */
export async function readCache(file) {
	try {
		const data = JSON.parse(await readFile(file, "utf8"));
		if (data?.version !== CACHE_VERSION || !Array.isArray(data.models) || data.models.length === 0) return undefined;
		return { checkedAt: data.checkedAt, models: data.models };
	} catch {
		return undefined;
	}
}

/** Atomically persist the catalog (mode 0600). Contains no secrets. */
export async function writeCache(file, models, checkedAt = Date.now()) {
	await mkdir(dirname(file), { recursive: true });
	const tmp = `${file}.${process.pid}.tmp`;
	await writeFile(tmp, `${JSON.stringify({ version: CACHE_VERSION, checkedAt, models })}\n`, { mode: 0o600 });
	await rename(tmp, file);
}
