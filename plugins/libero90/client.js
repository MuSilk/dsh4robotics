window.__ModuleLoader__.load({
  id: "@local/libero90",
  factory(require) {
    const React = require("react");
    const h = React.createElement;
    const { useEffect, useRef, useState } = React;
    const POS_KEY = "libero90-monitor-pos";
    const OPEN_KEY = "libero90-monitor-open";

    function readPosition() {
      try {
        const saved = JSON.parse(localStorage.getItem(POS_KEY) || "null");
        if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) return saved;
      } catch { /* keep the default corner */ }
      return { left: 16, top: Math.max(16, window.innerHeight - 220) };
    }

    function clamp(left, top, width, height) {
      const maxLeft = Math.max(8, window.innerWidth - width - 8);
      const maxTop = Math.max(8, window.innerHeight - height - 8);
      return {
        left: Math.min(Math.max(8, left), maxLeft),
        top: Math.min(Math.max(8, top), maxTop),
      };
    }

    function Monitor() {
      const [status, setStatus] = useState(null);
      const [tick, setTick] = useState(0);
      const [open, setOpen] = useState(() => localStorage.getItem(OPEN_KEY) !== "0");
      const [pos, setPos] = useState(readPosition);
      const drag = useRef(null);

      useEffect(() => {
        let stop = false;
        async function poll() {
          try {
            const response = await fetch("/libero90/status");
            if (!response.ok) throw new Error(String(response.status));
            const body = await response.json();
            if (!stop) setStatus(body);
          } catch {
            if (!stop) setStatus({ state: "offline" });
          }
        }
        poll();
        const timer = setInterval(() => {
          poll();
          setTick((value) => value + 1);
        }, 700);
        return () => {
          stop = true;
          clearInterval(timer);
        };
      }, []);

      function onPointerDown(event) {
        if (event.button !== 0 || event.target.closest("button")) return;
        const rect = event.currentTarget.parentElement.getBoundingClientRect();
        drag.current = {
          dx: event.clientX - rect.left,
          dy: event.clientY - rect.top,
          width: rect.width,
          height: rect.height,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
      }

      function onPointerMove(event) {
        if (!drag.current) return;
        const next = clamp(
          event.clientX - drag.current.dx,
          event.clientY - drag.current.dy,
          drag.current.width,
          drag.current.height,
        );
        setPos(next);
      }

      function onPointerUp() {
        if (!drag.current) return;
        drag.current = null;
        setPos((current) => {
          localStorage.setItem(POS_KEY, JSON.stringify(current));
          return current;
        });
      }

      function toggle() {
        setOpen((current) => {
          const next = !current;
          localStorage.setItem(OPEN_KEY, next ? "1" : "0");
          return next;
        });
      }

      const state = status?.state ?? "offline";
      const step = Number(status?.step ?? 0);
      const maxSteps = Number(status?.max_steps ?? 0);
      const width = maxSteps > 0 ? Math.min(100, Math.round((step / maxSteps) * 100)) : 0;
      const title = {
        idle: "空闲",
        running: "执行中",
        success: "成功",
        failed: "失败",
        error: "出错",
        offline: "未连接",
      }[state] ?? state;
      const panelStyle = {
        position: "fixed",
        left: `${pos.left}px`,
        top: `${pos.top}px`,
        width: open ? "280px" : "auto",
        zIndex: 30,
        pointerEvents: "auto",
        boxSizing: "border-box",
        borderRadius: "12px",
        background: "var(--dsw-alias-bg-elevated, var(--dsw-alias-bg, #17191e))",
        color: "var(--dsw-alias-fg, #f4f6f8)",
        border: "1px solid var(--dsw-alias-border, rgba(255,255,255,0.12))",
        boxShadow: "0 8px 28px rgba(0,0,0,0.28)",
        font: "13px/1.45 sans-serif",
      };
      const handleStyle = {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        cursor: "grab",
        userSelect: "none",
        touchAction: "none",
        padding: open ? "10px 10px 0" : "8px 10px",
        fontWeight: 650,
      };

      return h("aside", { "aria-label": "LIBERO-90 进度", style: panelStyle }, [
        h("div", {
          key: "handle",
          style: handleStyle,
          onPointerDown,
          onPointerMove,
          onPointerUp,
        }, [
          h("span", { key: "name", style: { flex: "1 1 auto" } }, open ? "LIBERO-90" : `LIBERO-90 · ${title}`),
          h("button", {
            key: "toggle",
            type: "button",
            onClick: toggle,
            style: {
              border: "0",
              borderRadius: "8px",
              padding: "2px 8px",
              cursor: "pointer",
              background: "var(--dsw-alias-bg-muted, rgba(255,255,255,0.08))",
              color: "inherit",
            },
          }, open ? "收起" : "展开"),
        ]),
        open ? h("div", { key: "body", style: { padding: "8px 12px 12px" } }, [
          h("div", { key: "state" }, `状态：${title}${status?.phase ? ` · ${{ idle: "空闲", interactive: "交互", formal: "正式评测", closed: "已提交" }[status.phase] ?? status.phase}` : ""}`),
          h("div", { key: "task", style: { marginTop: "4px" } }, status?.language || "还没有任务"),
          h("div", { key: "steps", style: { marginTop: "8px" } }, `步数 ${step} / ${maxSteps || "-"}`),
          h("div", {
            key: "track",
            style: {
              marginTop: "6px",
              height: "8px",
              borderRadius: "999px",
              background: "var(--dsw-alias-bg-muted, rgba(255,255,255,0.08))",
              overflow: "hidden",
            },
          }, h("div", {
            style: {
              width: `${width}%`,
              height: "100%",
              background: state === "success" ? "#2f9e62" : "#3b82f6",
            },
          })),
          status?.reason ? h("div", { key: "reason", style: { marginTop: "8px" } }, `结束原因：${status.reason}`) : null,
          status?.error ? h("div", { key: "error", style: { marginTop: "8px", color: "#f0a0a0" } }, status.error) : null,
          status?.frame ? h("img", {
            key: "frame",
            alt: "当前机械臂相机画面",
            src: `/libero90/frame.jpg?t=${tick}`,
            style: { display: "block", width: "100%", marginTop: "10px", borderRadius: "8px", background: "#000" },
          }) : null,
        ]) : null,
      ]);
    }

    return {
      inject: ["slots"],
      apply(ctx) {
        ctx.slots.inject("shell.overlay", () => ctx.slots.register({
          name: "shell.overlay",
          id: "libero90-monitor",
          order: 40,
        }, Monitor));
      },
    };
  },
});
