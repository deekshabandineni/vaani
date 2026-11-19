// Real-time cross-device sync over WebSockets.
// Each authenticated socket belongs to a user. Events from one device are
// relayed to that user's other devices: new attempts, updated phoneme stats,
// and "session" state (which drill is open) so practice can move from phone
// to laptop mid-session.
import { WebSocketServer } from "ws";

export class Hub {
  constructor() { this.sockets = new Map(); } // userId -> Set<ws>

  add(userId, ws) {
    if (!this.sockets.has(userId)) this.sockets.set(userId, new Set());
    this.sockets.get(userId).add(ws);
    ws.on("close", () => {
      const set = this.sockets.get(userId);
      set?.delete(ws);
      if (set && set.size === 0) this.sockets.delete(userId);
    });
  }

  count(userId) { return this.sockets.get(userId)?.size ?? 0; }

  broadcast(userId, message, exceptDeviceId = null) {
    const data = JSON.stringify(message);
    for (const ws of this.sockets.get(userId) ?? []) {
      if (ws.readyState === 1 && ws.deviceId !== exceptDeviceId) ws.send(data);
    }
  }
}

export function attachWebSocket(server, hub, verifyToken) {
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (req, socket, head) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname !== "/ws") return socket.destroy();
    let claims;
    try { claims = verifyToken(url.searchParams.get("token") ?? ""); } catch {
      socket.write("HTTP/1.1 401 Unauthorized\r\n\r\n");
      return socket.destroy();
    }
    wss.handleUpgrade(req, socket, head, (ws) => {
      ws.deviceId = url.searchParams.get("device") || "unknown";
      hub.add(claims.sub, ws);
      ws.send(JSON.stringify({ type: "hello", devices: hub.count(claims.sub) }));
      hub.broadcast(claims.sub, { type: "presence", devices: hub.count(claims.sub) }, ws.deviceId);
      ws.on("message", (raw) => {
        let msg;
        try { msg = JSON.parse(raw.toString()); } catch { return; }
        if (msg?.type === "session" && typeof msg.state === "object") {
          hub.broadcast(claims.sub, { type: "session", state: msg.state, from: ws.deviceId }, ws.deviceId);
        }
      });
    });
  });
  return wss;
}
