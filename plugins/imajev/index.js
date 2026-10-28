import { connect } from "node:net";
import { defineTool } from "@deepseek-ai/dsh-tools";

const OUTPUT = {
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      text: { type: "string", required: true },
    },
  },
  render(_args, value) {
    return [{ type: "text", text: value.text }];
  },
};

const socketPath = process.env.IMAJEV_SOCKET ?? "";

let socket = null;
let buffer = "";
let seq = 0;
let chain = Promise.resolve();
let starting = null;
const pending = new Map();

function onData(chunk) {
  buffer += chunk;
  let newline = buffer.indexOf("\n");
  while (newline >= 0) {
    const line = buffer.slice(0, newline);
    buffer = buffer.slice(newline + 1);
    newline = buffer.indexOf("\n");
    if (!line.trim()) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    const waiter = pending.get(message.id);
    if (!waiter) continue;
    pending.delete(message.id);
    waiter.resolve(message);
  }
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    if (!socketPath) {
      reject(new Error("IMAJEV_SOCKET is not set"));
      return;
    }
    const next = connect(socketPath);
    const timer = setTimeout(() => {
      next.destroy();
      reject(new Error("imajev socket timeout"));
    }, 5000);
    next.once("connect", () => {
      clearTimeout(timer);
      socket = next;
      buffer = "";
      socket.setEncoding("utf8");
      socket.on("data", onData);
      socket.on("error", () => {});
      socket.on("close", () => {
        socket = null;
        for (const waiter of pending.values()) waiter.reject(new Error("imajev disconnected"));
        pending.clear();
      });
      resolve();
    });
    next.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function request(method, params) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
  });
}

async function ensureReady() {
  if (socket && !socket.destroyed) return;
  if (!starting) {
    starting = (async () => {
      const deadline = Date.now() + 30000;
      let last = new Error(`imajev is not listening at ${socketPath || "(unset)"}`);
      while (Date.now() < deadline) {
        try {
          await connectSocket();
          const init = await request("initialize", {
            protocolVersion: "2025-06-18",
            capabilities: {},
            clientInfo: { name: "dsh-plugin", version: "1" },
          });
          if (init.error) throw new Error(init.error.message || "initialize failed");
          return;
        } catch (error) {
          last = error;
          if (socket) socket.destroy();
          socket = null;
          await new Promise((resolve) => setTimeout(resolve, 300));
        }
      }
      throw last;
    })().finally(() => {
      starting = null;
    });
  }
  await starting;
}

function rpc(method, params) {
  const run = async () => {
    await ensureReady();
    return request(method, params);
  };
  const next = chain.then(run, run);
  chain = next.then(() => {}, () => {});
  return next;
}

export const inject = ["tools"];

export function apply(ctx) {
  ctx.effect(() => {
    const dispose = ctx.tools.register(defineTool({
      name: "imajev_decide",
      description: "Score 1-8 questions about 0-2 pictures with Imajev-4B. Does not move the arm and does not measure depth. images entries are agentview, wrist, or a workspace-relative JPEG/PNG/WebP path (experiments/data/frame_probe0.jpg). At most two; agentview and wrist are the current simulator frames. questions is a Jev object: each value has type noul, choice, score, or multi, plus instructions and criteria. state is an optional JSON object the instructions can name with backticks; omit it for a picture-only question. Read unknown_probability and abstained before using an answer. Abstention means the picture does not show it.",
      timeoutMs: 4 * 60 * 1000,
      parameters: {
        questions: {
          type: "json",
          required: true,
          description: "Jev questions object, 1 to 8 entries. Example: {\"which\": {\"type\": \"choice\", \"instructions\": \"Which team should handle this?\", \"criteria\": {\"billing\": \"a charge or invoice\", \"other\": \"anything else\"}}}",
        },
        images: {
          type: "array",
          description: "Zero to two of: agentview, wrist, or a workspace-relative image path.",
          items: { type: "string" },
        },
        state: {
          type: "json",
          description: "Optional JSON value. A string or object. Omit it when the question is only about the pictures.",
        },
      },
      output: OUTPUT,
      async execute(args) {
        const response = await rpc("tools/call", { name: "imajev_decide", arguments: args ?? {} });
        if (response.error) throw new Error(response.error.message || "imajev_decide failed");
        const text = response.result?.content?.[0]?.text ?? "";
        return { text };
      },
    }));
    return () => {
      dispose();
      if (socket) socket.destroy();
    };
  });
}
