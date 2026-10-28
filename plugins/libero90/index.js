import { connect } from "node:net";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
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

export const TOOLS = [
  {
    name: "status",
    timeoutMs: 60 * 1000,
    description: "Task, episode state, and step limit.",
    parameters: {},
  },
  {
    name: "register",
    timeoutMs: 60 * 1000,
    description: "Freeze a controller.py bundle without running it. source is a project under code/ in the workspace, for example code/close_drawer. main(ctx) must use only the ctx it is given. A controller may call only status, libero_status, libero_observe, libero_cameras, libero_depth, libero_step, and libero_hold. ctx.call returns the MCP result; the JSON payload is json.loads(result['content'][0]['text']).",
    parameters: {
      source: {
        type: "string",
        required: true,
        description: "Project under code/ in the workspace, for example code/close_drawer.",
      },
    },
  },
  {
    name: "finish",
    timeoutMs: 60 * 1000,
    summarize: true,
    description: "End the episode and close it. Does not open another episode and does not run a controller.",
    parameters: {},
  },
  {
    name: "libero_status",
    timeoutMs: 3 * 60 * 1000,
    description: "Episode status, step budget, official LIBERO action and state layouts, cameras and instruction. No physics. Read this before composing actions. Actions are 7-D normalized OSC_POSE deltas in [-1, 1], not absolute poses.",
    parameters: {},
  },
  {
    name: "libero_observe",
    timeoutMs: 60 * 1000,
    description: "Current end-effector pose, gripper, joints, instruction, and camera names. No physics and no image bytes. Use libero_cameras for the pictures.",
    parameters: {},
  },
  {
    name: "libero_cameras",
    timeoutMs: 60 * 1000,
    description: "Agentview and wrist pictures for the current scene, with the end-effector pose, gripper, joints, and instruction. Does not move the arm. agentview_jpeg and wrist_jpeg are base64 JPEG, row 0 at the top, 512 by 512. Decode them in code; do not print the images.",
    parameters: {},
  },
  {
    name: "libero_depth",
    timeoutMs: 60 * 1000,
    description: "Agentview depth for the current scene, aligned with frame.jpg. Does not move the arm. depth_mm is base64 little-endian uint16 millimetres, row 0 at the top. camera_to_base maps camera-frame points into the same frame as eef_pos. Unproject in code; do not print the map.",
    parameters: {},
  },
  {
    name: "libero_step",
    timeoutMs: 3 * 60 * 1000,
    description: "Execute 1..50 official LIBERO OSC_POSE rows, one 20 Hz environment step each. Each row is 7 numbers in [-1, 1]: dx, dy, dz, droll, dpitch, dyaw, gripper. Gripper is negative to open and positive to close. At +/-1, translation is 0.05 m and rotation is 0.5 rad. Stops early only when the episode step limit is reached. Later rows of that call are not run.",
    parameters: {
      actions: {
        type: "array",
        required: true,
        description: "1..50 rows. Each row is 7 numbers in [-1, 1].",
        items: { type: "array", items: { type: "number" } },
      },
    },
  },
  {
    name: "libero_hold",
    timeoutMs: 3 * 60 * 1000,
    description: "Repeat the previous action row for 1..50 environment steps. Before any step, the hold row is zero motion with the gripper open. Each repeated row counts against the episode step limit.",
    parameters: {
      steps: {
        type: "integer",
        required: true,
        description: "How many environment steps to repeat, from 1 to 50.",
      },
    },
  },
];

export const inject = ["tools"];

function episodeOver(value) {
  return Boolean(value.session_episode_over || value.episode_ended);
}

function followUp(text) {
  let value = {};
  try {
    value = JSON.parse(text);
  } catch { /* keep the raw result in the prompt */ }
  let lead;
  if (value.finished) {
    lead = "This episode has ended.";
  } else if (episodeOver(value)) {
    lead = "This episode has hit its limit before finish.";
  } else {
    lead = "Follow SKILL.md.";
  }
  if (value.task_complete === true) {
    lead += " The task predicate is satisfied.";
  } else if (value.task_complete === false) {
    const failed = Array.isArray(value.failure_reason) ? value.failure_reason.filter(Boolean) : [];
    lead += failed.length
      ? ` The task predicate is not satisfied: ${failed.join("; ")}.`
      : " The task predicate is not satisfied.";
  }
  return [
    lead,
    "What do you know now that you wish you had known at the start of this episode? Concrete, transferable facts about this rig, task, or embodiment (camera mounting and extrinsics, table and base geometry, gripper axis and offsets, controller behavior, metric scale), written as advice to a future agent attempting the same task. Say 'none' if nothing qualifies.",
    text,
  ].join("\n");
}

export function apply(ctx, config = {}) {
  const runDir = config.runDir ?? "/data/changzou/deepseek4robotics/runs/libero90";
  const socketPath = process.env.LIBERO_SOCKET ?? `${runDir}/sock/mcp.sock`;

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
      const next = connect(socketPath);
      const timer = setTimeout(() => {
        next.destroy();
        reject(new Error("worker socket timeout"));
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
          for (const waiter of pending.values()) waiter.reject(new Error("worker disconnected"));
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
        let last = new Error(`simulation server is not listening at ${socketPath}`);
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

  ctx.effect(() => {
    const dispose = TOOLS.map((spec) => ctx.tools.register(defineTool({
      name: spec.name,
      description: spec.description,
      parameters: spec.parameters,
      output: OUTPUT,
      timeoutMs: spec.timeoutMs,
      async execute(args, exec) {
        const response = await rpc("tools/call", { name: spec.name, arguments: args ?? {} });
        if (response.error) throw new Error(response.error.message || `${spec.name} failed`);
        const text = response.result?.content?.[0]?.text ?? "";
        let ended = false;
        try {
          ended = episodeOver(JSON.parse(text));
        } catch { /* a non-JSON result does not end the episode */ }
        if (spec.summarize || ended) {
          exec.deferContext(createUserMessage({
            source: { kind: "user" },
            content: [{ type: "text", text: followUp(text) }],
          }));
        }
        return { text };
      },
    })));
    return () => {
      for (const stop of dispose) stop();
      if (socket) socket.destroy();
    };
  });
}
