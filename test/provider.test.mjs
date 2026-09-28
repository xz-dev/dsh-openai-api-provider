import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { apply } from "../index.js";
import { mapCatalog, mapRow, readCache, writeCache } from "../lib/catalog.js";

const KEY = "sk-test-secret-123";

test("mapRow: reasoning levels, minimal aliases low, none-only is non-reasoning", () => {
	const r = mapRow({ id: "a", supported_reasoning_levels: [{ effort: "low" }, "medium", "max", "bogus"] });
	assert.deepEqual(r.thinkingLevelMap, { off: null, minimal: "low", low: "low", medium: "medium", high: null, xhigh: null, max: "max" });
	assert.equal(mapRow({ id: "b", supported_reasoning_levels: ["none"] }).thinkingLevelMap, undefined);
	assert.deepEqual(mapRow({ id: "c", capabilities: { effort_tiers: ["high"] } }).thinkingLevelMap.high, "high");
});

test("mapRow: limits, modalities, and non-chat filtering", () => {
	const r = mapRow({ id: "a", name: "A", context_window: 262000, max_output_tokens: 64000, input_modalities: ["text", "image"] });
	assert.deepEqual(r, { id: "a", name: "A", contextWindow: 262000, maxTokens: 64000, input: ["text", "image"] });
	assert.equal(mapRow({ id: "d" }).contextWindow, 128000);
	assert.equal(mapRow({ id: "e", context_length: 1000, max_tokens: 1000 }).maxTokens, undefined, "echoed cap is dropped");
	assert.equal(mapRow({ id: "text-embedding-3" }), undefined);
	assert.equal(mapRow({ id: "f", type: "image" }), undefined);
	assert.equal(mapRow({ id: "g", output_modalities: ["audio"] }), undefined);
	assert.equal(mapRow({ name: "no id" }), undefined);
});

test("mapCatalog: sorted, de-duplicated, empty rejected", () => {
	assert.deepEqual(mapCatalog([{ id: "b" }, { id: "a" }, { id: "b" }]).map((m) => m.id), ["a", "b"]);
	assert.throws(() => mapCatalog([{ id: "x-embedding" }]), /no usable models/);
});

test("cache round-trip is private and rejects foreign data", async () => {
	const dir = await mkdtemp(join(tmpdir(), "oap-"));
	try {
		const file = join(dir, "sub", "c.json");
		await writeCache(file, [{ id: "a", name: "a", contextWindow: 1, input: ["text"] }], 5);
		assert.equal((await stat(file)).mode & 0o777, 0o600);
		assert.deepEqual((await readCache(file)).checkedAt, 5);
		assert.equal(await readCache(join(dir, "missing.json")), undefined);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

async function stubGateway() {
	const state = { rows: [{ id: "m-reason", name: "M Reason", context_window: 1050000, max_output_tokens: 128000, input_modalities: ["text", "image"], supported_reasoning_levels: ["low", "medium", "high"] }, { id: "m-embedding" }], status: 200, auths: [] };
	const server = createServer((req, res) => {
		state.auths.push(req.headers.authorization);
		state.clientVersion = new URL(req.url, "http://x").searchParams.get("client_version");
		if (state.status !== 200) {
			res.writeHead(state.status, { "content-type": "text/plain" });
			return res.end(`denied for ${req.headers.authorization}`);
		}
		res.writeHead(200, { "content-type": "application/json" });
		res.end(JSON.stringify({ models: state.rows }));
	});
	await new Promise((r) => server.listen(0, "127.0.0.1", r));
	return { state, url: `http://127.0.0.1:${server.address().port}/v1`, close: () => new Promise((r) => server.close(r)) };
}

function fakeCtx(env) {
	const events = { replace: 0, registered: undefined, commands: new Map(), warnings: [], disposers: [] };
	const ctx = {
		logger: () => ({ warn: (m) => events.warnings.push(m), info() {}, error() {} }),
		effect: (fn) => {
			const d = fn();
			if (typeof d === "function") events.disposers.push(d);
		},
		inject: (_names, fn) => fn(ctx),
		get: (name) => (name === "launchEnvironment" ? { get: (k) => (env[k] === undefined ? undefined : { value: env[k] }) } : undefined),
		commands: { register: (def) => events.commands.set(def.name, def) },
		llm: {
			registerAdapter(routes, adapter) {
				events.registered = { routes, adapter };
				const handle = () => {};
				handle.replace = () => void events.replace++;
				return handle;
			},
		},
	};
	return { ctx, events, dispose: () => events.disposers.forEach((d) => d()) };
}

test("startup without cache fetches, registers the route, and describes models", async () => {
	const gw = await stubGateway();
	const dir = await mkdtemp(join(tmpdir(), "oap-"));
	const { ctx, events, dispose } = fakeCtx({ OPENAI_API_EXTENSION_BASE_URL: gw.url, OPENAI_API_EXTENSION_API_KEY: KEY });
	try {
		await apply(ctx, { cacheFile: join(dir, "c.json") });
		const { routes, adapter } = events.registered;
		assert.deepEqual(routes, ["gateway"]);
		// model operations wait for the first fetch (gate), so this sees the catalog
		const listed = await adapter.listModels("gateway");
		assert.deepEqual(listed.map((m) => m.id), ["m-reason"]);
		assert.equal(gw.state.auths[0], `Bearer ${KEY}`);
		assert.equal(gw.state.clientVersion, "0.1.0", "client_version selects the full gateway catalog");
		assert.deepEqual((await adapter.listModels("gateway")).map((m) => m.id), ["m-reason"]);
		const info = await adapter.resolveModel("gateway", "m-reason");
		assert.equal(info.context.contextWindow, 1050000);
		assert.deepEqual(info.reasoning.efforts.map((e) => e.id), ["minimal", "low", "medium", "high"]);
		assert.equal(adapter.providerRetryPolicy("gateway").mode, "normal");
		assert.ok(!(await readFile(join(dir, "c.json"), "utf8")).includes(KEY), "cache holds no secret");
	} finally {
		dispose();
		await gw.close();
		await rm(dir, { recursive: true, force: true });
	}
});

test("cached catalog serves at once; /gateway refresh swaps models and re-announces", async () => {
	const gw = await stubGateway();
	const dir = await mkdtemp(join(tmpdir(), "oap-"));
	const cacheFile = join(dir, "c.json");
	await writeCache(cacheFile, [{ id: "cached", name: "Cached", contextWindow: 1000, input: ["text"] }], 1);
	gw.state.status = 503;
	const { ctx, events, dispose } = fakeCtx({ OPENAI_API_EXTENSION_BASE_URL: gw.url, OPENAI_API_EXTENSION_API_KEY: KEY });
	try {
		await apply(ctx, { cacheFile });
		const { adapter } = events.registered;
		await adapter.ready;
		assert.deepEqual((await adapter.listModels("gateway")).map((m) => m.id), ["cached"]);
		const cmd = events.commands.get("gateway");
		// background refresh fails (503) → cached catalog kept, error redacted
		let r = await cmd.handler({ rawInput: "refresh" });
		assert.equal(r.kind, "error");
		assert.ok(!r.text.includes(KEY), "key redacted from error");
		assert.match(r.text, /HTTP 503/);
		assert.deepEqual((await adapter.listModels("gateway")).map((m) => m.id), ["cached"]);
		gw.state.status = 200;
		const before = events.replace;
		r = await cmd.handler({ rawInput: "refresh" });
		assert.equal(r.kind, "success");
		assert.match(r.text, /Updated: 1 models/);
		assert.equal(events.replace, before + 1);
		assert.deepEqual((await adapter.listModels("gateway")).map((m) => m.id), ["m-reason"]);
		r = await cmd.handler({ rawInput: "refresh" });
		assert.match(r.text, /Unchanged/);
		assert.equal(events.replace, before + 1, "unchanged catalog does not re-announce");
		r = await cmd.handler({ rawInput: "" });
		assert.match(r.text, /Models: 1/);
		assert.ok(!r.text.includes(gw.url), "status does not print the server URL");
	} finally {
		dispose();
		await gw.close();
		await rm(dir, { recursive: true, force: true });
	}
});

test("missing key fails the call with MISSING_CREDENTIAL; missing base URL registers an empty route", async () => {
	const dir = await mkdtemp(join(tmpdir(), "oap-"));
	try {
		const a = fakeCtx({});
		await apply(a.ctx, { cacheFile: join(dir, "a.json") });
		await a.events.registered.adapter.ready;
		assert.deepEqual(await a.events.registered.adapter.listModels("gateway"), []);
		assert.match(a.events.warnings.join("\n"), /no usable base URL/);
		a.dispose();

		const gw = await stubGateway();
		const b = fakeCtx({ OPENAI_API_EXTENSION_BASE_URL: gw.url });
		await apply(b.ctx, { cacheFile: join(dir, "b.json") });
		await b.events.registered.adapter.ready;
		assert.equal(gw.state.auths.length, 0, "no unauthenticated request");
		assert.match(b.events.warnings.join("\n"), /no credential/);
		b.dispose();
		await gw.close();
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});
