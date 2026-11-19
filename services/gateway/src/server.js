import { readFileSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import jwt from "jsonwebtoken";
import pg from "pg";

import { createApp } from "./app.js";
import { createDb } from "./db.js";
import { attachWebSocket, Hub } from "./hub.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const sharedDir = process.env.SHARED_DIR ?? path.resolve(here, "../../../shared");
const readJson = (f) => JSON.parse(readFileSync(path.join(sharedDir, f), "utf8"));

const jwtSecret = process.env.JWT_SECRET;
if (!jwtSecret || jwtSecret.length < 16) {
  console.error("JWT_SECRET must be set (16+ characters)");
  process.exit(1);
}

const dbUrl = process.env.DATABASE_URL ?? "postgres://vaani:vaani@localhost:5432/vaani";
let pool;
if (dbUrl === "memory") {
  // quick demos and e2e tests without Postgres (needs the pg-mem dev dependency; data is lost on restart)
  const { newDb } = await import("pg-mem");
  const { Pool } = newDb().adapters.createPg();
  pool = new Pool();
  console.warn("using in-memory database: data will not persist");
} else {
  pool = new pg.Pool({ connectionString: dbUrl });
}
const db = createDb(pool);
for (let i = 0; ; i++) {
  try { await db.migrate(); break; } catch (err) {
    if (i >= 15) throw err;
    console.log("waiting for database...", err.code ?? err.message);
    await new Promise((r) => setTimeout(r, 2000));
  }
}

const hub = new Hub();
const app = createApp({ db, hub, drills: readJson("drills.json"), profiles: readJson("l1_profiles.json"),
                        jwtSecret, mlApiUrl: process.env.ML_API_URL ?? "http://localhost:8000" });
const server = http.createServer(app);
attachWebSocket(server, hub, (token) => jwt.verify(token, jwtSecret));
const port = Number(process.env.PORT ?? 3000);
server.listen(port, () => console.log(`gateway listening on :${port}`));
