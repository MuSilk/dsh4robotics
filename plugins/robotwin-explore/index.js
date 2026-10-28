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
    description: "Task, this round, the action type, and the exploration budget. The seed is not included.",
    parameters: {},
  },
  {
    name: "score",
    timeoutMs: 60 * 1000,
    description: "Success predicate for the current round. Does not move the arms and does not spend budget. A controller cannot call this.",
    parameters: {},
  },
  {
    name: "reset",
    timeoutMs: 15 * 60 * 1000,
    description: "Score this round, close it, and open a new random seed. Takes no arguments. When the session covers every RoboTwin task, the task can change too; read the instruction. Fails when the exploration budget is spent. Does not itself spend actions.",
    parameters: {},
  },
  {
    name: "run",
    timeoutMs: 70 * 60 * 1000,
    description: "Run code/<name>/controller.py main(ctx) on the current round. Each planned action counts against the exploration budget and this round's cap. Does not reset. The controller cannot call score, reset, run, or submit, and it does not receive the budget or the seed.",
    parameters: {
      source: {
        type: "string",
        required: true,
        description: "Project under code/, for example code/place. controller.py must define main(ctx).",
      },
    },
  },
  {
    name: "submit",
    timeoutMs: 60 * 1000,
    summarize: true,
    description: "Snapshot code/ and roll that controller out on held-out seeds, with no agent in the loop. Ends exploration. Scores are not returned.",
    parameters: {
      source: {
        type: "string",
        required: true,
        description: "Project under code/ to freeze, for example code/place.",
      },
    },
  },
  {
    name: "robotwin_status",
    timeoutMs: 3 * 60 * 1000,
    description: "Instruction, action layout, cameras, and this round's remaining actions. No physics. An action is an absolute dual-arm target, not a delta. ee is 16 numbers: left xyz, left quaternion wxyz, left gripper, right xyz, right quaternion wxyz, right gripper. qpos is left joints, left gripper, right joints, right gripper. Gripper 0 is closed and 1 is open.",
    parameters: {},
  },
  {
    name: "robotwin_observe",
    timeoutMs: 60 * 1000,
    description: "Current left and right end-effector poses, grippers, joints, instruction, and camera names. No image bytes. Use robotwin_cameras for the pictures.",
    parameters: {},
  },
  {
    name: "robotwin_cameras",
    timeoutMs: 3 * 60 * 1000,
    description: "Head, left-wrist, and right-wrist pictures for the current scene, with both end-effector poses. Does not move the arms. head_jpeg, left_jpeg, and right_jpeg are base64 JPEG, row 0 at the top. Decode them in code; do not print the images. imajev calls the head frame agentview and the left wrist wrist.",
    parameters: {},
  },
  {
    name: "robotwin_depth",
    timeoutMs: 3 * 60 * 1000,
    description: "Head-camera depth for the current scene, aligned with the head JPEG. Does not move the arms. depth_mm is base64 little-endian uint16 millimetres, row 0 at the top. fx, fy, cx, cy are the pinhole intrinsics. cam2world_gl is SAPIEN's camera matrix, 16 numbers, row-major. Unproject in code; do not print the map.",
    parameters: {},
  },
  {
    name: "robotwin_step",
    timeoutMs: 15 * 60 * 1000,
    description: "Execute 1..8 official RoboTwin actions. Each row is one take_action: a motion plan from the current arm state to that absolute target, then the execution. It is not a 20 Hz servo tick. A plan that fails still spends one action and the arm may not move. ee rows have 16 numbers in the layout from robotwin_status. qpos rows have action_dim numbers. Gripper values are in [0, 1]. Stops early when this round's cap or the exploration budget is reached.",
    parameters: {
      actions: {
        type: "array",
        required: true,
        description: "1..8 rows. ee: 16 numbers. qpos: action_dim numbers from robotwin_status.",
        items: { type: "array", items: { type: "number" } },
      },
    },
  },
  {
    name: "robotwin_hold",
    timeoutMs: 15 * 60 * 1000,
    description: "Repeat the previous action row for 1..8 planned actions. Before any step, the hold row is the current pose with the grippers left as they are. Each repeated row counts against the budget.",
    parameters: {
      steps: {
        type: "integer",
        required: true,
        description: "How many planned actions to repeat, from 1 to 8.",
      },
    },
  },
];

export const inject = ["tools"];

function followUp(text) {
  let value = {};
  try {
    value = JSON.parse(text);
  } catch { /* keep the raw result */ }
  if (value.submitted) {
    return [
      "Submitted. Held-out evaluation is running without you, and further robot calls will fail.",
      "What must the controller recompute on every new seed, instead of hard-coding from one round? Concrete facts only. Say none if nothing qualifies.",
      text,
    ].join("\n");
  }
  if (value.reason === "budget") {
    return `The exploration budget is spent. submit the controller.\n${text}`;
  }
  if (value.episode_ended) {
    return `This round stopped (${value.reason || "ended"}). reset opens a new random seed while budget remains. Read the instruction again.\n${text}`;
  }
  return text;
}

export function apply(ctx) {
  const socketPath = process.env.ROBOTWIN_SOCKET || process.env.LIBERO_SOCKET;
  if (!socketPath) throw new Error("ROBOTWIN_SOCKET is not set");

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
        let follow = false;
        try {
          const value = JSON.parse(text);
          follow = Boolean(value.submitted || value.episode_ended || value.reason === "budget");
        } catch { /* a non-JSON result does not end the round */ }
        if (spec.summarize || follow) {
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
