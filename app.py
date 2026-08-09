from __future__ import annotations

import html
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import gradio as gr

from fixed.config import CONFIG, STATIC_DIR
from fixed.agent_runtime import AgentRuntime


runtime = AgentRuntime()
CSS_PATH = STATIC_DIR / "app.css"
DEFAULT_PENDING_STATUS = "답변을 진행중입니다"
ENTER_TO_SEND_HEAD = """
<style>
@media (min-width: 981px) {
  html,
  body {
    height: 100dvh !important;
    overflow-y: hidden !important;
  }
}
</style>
<script>
function setKananaTextareaValue(textarea, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
  setter.call(textarea, value);
  textarea.dispatchEvent(new Event("input", { bubbles: true }));
  textarea.dispatchEvent(new Event("change", { bubbles: true }));
}

document.addEventListener("keydown", function(event) {
  const target = event.target;
  if (!(target instanceof HTMLTextAreaElement)) return;
  if (!target.closest("#kanana-input")) return;
  if (event.key !== "Enter" || event.isComposing) return;

  event.preventDefault();
  event.stopPropagation();

  if (event.shiftKey) {
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? target.value.length;
    const nextValue = `${target.value.slice(0, start)}\n${target.value.slice(end)}`;
    setKananaTextareaValue(target, nextValue);
    target.setSelectionRange(start + 1, start + 1);
    return;
  }

  const sendRoot = document.querySelector("#kanana-send");
  const sendButton =
    sendRoot?.matches("button") ? sendRoot : sendRoot?.querySelector("button");

  if (sendButton && !sendButton.disabled) {
    sendButton.dispatchEvent(new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
      view: window
    }));
  }
}, true);

(function setupKananaUiControls() {
  if (window.__kananaUiControlsReady) return;
  window.__kananaUiControlsReady = true;
  const WIDTH_KEY = "kanana.conversationDrawer.width.v2";
  const COLLAPSED_KEY = "kanana.conversationDrawer.collapsed.v2";
  const SCHEDULE_WIDTH_KEY = "kanana.schedulePanel.widthPercent";

  function applyDrawerState(collapsed, width) {
    const shell = document.querySelector("#kanana-shell");
    const drawer = document.querySelector("#conversation-drawer");
    const toggle = document.querySelector("#conversation-drawer-toggle");
    if (!shell || !drawer || !toggle) return false;
    const safeWidth = Math.max(200, Math.min(360, Number(width) || 240));
    drawer.style.setProperty("--conversation-drawer-width", `${safeWidth}px`);
    shell.classList.toggle("conversation-drawer-collapsed", collapsed);
    toggle.classList.toggle("is-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", collapsed ? "대화 내역 펼치기" : "대화 내역 접기");
    toggle.setAttribute("title", collapsed ? "대화 내역 펼치기" : "대화 내역 접기");
    return true;
  }

  document.addEventListener("click", (event) => {
    const toggle = event.target.closest("#conversation-drawer-toggle");
    if (!toggle) return;
    const collapsed = !document.querySelector("#kanana-shell")?.classList.contains("conversation-drawer-collapsed");
    localStorage.setItem(COLLAPSED_KEY, String(collapsed));
    applyDrawerState(collapsed, localStorage.getItem(WIDTH_KEY));
  });

  document.addEventListener("pointerdown", (event) => {
    const resizer = event.target.closest(".saved-field-resizer, #conversation-drawer-resizer, #schedule-panel-resizer");
    if (!resizer || window.matchMedia("(max-width: 980px)").matches) return;
    event.preventDefault();

    if (resizer.id === "conversation-drawer-resizer") {
      const drawer = document.querySelector("#conversation-drawer");
      const shell = document.querySelector("#kanana-shell");
      if (!drawer || shell?.classList.contains("conversation-drawer-collapsed")) return;
      const startX = event.clientX;
      const startWidth = drawer.getBoundingClientRect().width;
      document.body.classList.add("resizing-conversation-drawer");
      const move = (moveEvent) => {
        const width = Math.max(200, Math.min(360, startWidth + moveEvent.clientX - startX));
        applyDrawerState(false, width);
      };
      const end = (endEvent) => {
        const width = drawer.getBoundingClientRect().width;
        localStorage.setItem(WIDTH_KEY, String(width));
        document.body.classList.remove("resizing-conversation-drawer");
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", end);
        document.removeEventListener("pointercancel", end);
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", end);
      document.addEventListener("pointercancel", end);
      return;
    }

    if (resizer.id === "schedule-panel-resizer") {
      const shell = document.querySelector("#kanana-shell");
      const schedulePanel = document.querySelector(".schedule-sidebar");
      if (!shell || !schedulePanel) return;
      const startX = event.clientX;
      const startWidth = schedulePanel.getBoundingClientRect().width;
      const shellWidth = shell.getBoundingClientRect().width;
      document.body.classList.add("resizing-schedule-panel");
      const move = (moveEvent) => {
        const maxWidth = Math.min(620, shellWidth * 0.58);
        const nextWidth = Math.max(260, Math.min(maxWidth, startWidth + moveEvent.clientX - startX));
        const widthPercent = (nextWidth / shellWidth) * 100;
        schedulePanel.style.setProperty("--schedule-panel-width", `${widthPercent}%`);
      };
      const end = () => {
        const widthPercent = (schedulePanel.getBoundingClientRect().width / shellWidth) * 100;
        localStorage.setItem(SCHEDULE_WIDTH_KEY, String(widthPercent));
        document.body.classList.remove("resizing-schedule-panel");
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", end);
        document.removeEventListener("pointercancel", end);
      };
      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", end);
      document.addEventListener("pointercancel", end);
      return;
    }

    const before = document.getElementById(resizer.dataset.resizeBefore);
    const after = document.getElementById(resizer.dataset.resizeAfter);
    if (!before || !after) return;
    const startY = event.clientY;
    const beforeHeight = before.getBoundingClientRect().height;
    const totalHeight = beforeHeight + after.getBoundingClientRect().height;
    resizer.classList.add("is-dragging");
    document.body.classList.add("resizing-saved-field");
    const move = (moveEvent) => {
      const nextBefore = Math.max(100, Math.min(totalHeight - 100, beforeHeight + moveEvent.clientY - startY));
      before.style.flex = `0 0 ${nextBefore}px`;
      after.style.flex = `0 0 ${totalHeight - nextBefore}px`;
    };
    const end = () => {
      resizer.classList.remove("is-dragging");
      document.body.classList.remove("resizing-saved-field");
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", end);
      document.removeEventListener("pointercancel", end);
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", end);
    document.addEventListener("pointercancel", end);
  });

  const restoreLayout = () => {
    const savedCollapsed = localStorage.getItem(COLLAPSED_KEY);
    applyDrawerState(savedCollapsed === null ? true : savedCollapsed === "true", localStorage.getItem(WIDTH_KEY));
    const savedScheduleWidth = Number(localStorage.getItem(SCHEDULE_WIDTH_KEY));
    const schedulePanel = document.querySelector(".schedule-sidebar");
    if (schedulePanel && Number.isFinite(savedScheduleWidth) && savedScheduleWidth > 0) {
      schedulePanel.style.setProperty("--schedule-panel-width", `${Math.max(22, Math.min(58, savedScheduleWidth))}%`);
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", restoreLayout, { once: true });
  else setTimeout(restoreLayout, 0);
})();

</script>
"""
DELETE_CONVERSATION_CONFIRM_JS = """
function(conversationId) {
  if (!conversationId) {
    alert("삭제할 저장된 대화를 먼저 선택해 주세요.");
    return [""];
  }
  const confirmed = confirm("선택한 저장된 대화를 영구 삭제할까요?\\n이 작업은 되돌릴 수 없습니다.");
  return [confirmed ? conversationId : ""];
}
"""


def _sqlite_schedule_memory_enabled() -> bool:
    return int(getattr(runtime, "active_week", CONFIG.active_week) or 1) >= 3


def _saved_item_tables(
    limit: int = 50,
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    if not _sqlite_schedule_memory_enabled():
        return [], [], []

    schedules = runtime.app_store.list_schedules(limit=limit)
    reminders = runtime.app_store.list_reminders(limit=limit)
    todos = runtime.app_store.list_todos(limit=limit)

    schedule_rows = [
        [
            row.get("date") or "날짜 미정",
            row.get("start_time") or "시간 미정",
            row.get("end_time") or "",
            row.get("title") or "제목 없음",
            ", ".join(row.get("attendees") or []),
        ] for row in schedules
    ]

    reminder_rows = [
        [
            row.get("date") or "날짜 미정",
            row.get("start_time") or "시간 미정",
            row.get("title") or "제목 없음",
        ] for row in reminders
    ]

    todo_rows = [
        [
            row.get("priority"),
            row.get("due_date") or "날짜 미정",
            row.get("end_time") or "시간 미정",
            row.get("title") or "제목 없음",
        ] for row in todos
    ]

    return schedule_rows, reminder_rows, todo_rows


def _chat_notice() -> list[dict[str, str]]:
    return []


def _pending_assistant_message(status_text: str = DEFAULT_PENDING_STATUS) -> dict[str, str]:
    return {"role": "assistant", "content": f"...\n\n<small>{html.escape(status_text)}</small>"}


def _message_field(message: Any, field: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(field)
    return getattr(message, field, None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        return "\n".join(
            text for content_block in content
            if (text := _content_text(content_block))
        )

    if isinstance(content, Mapping):
        content_type = content.get("type")
        text = content.get("text")
    else:
        content_type = getattr(content, "type", None)
        text = getattr(content, "text", None)

    if content_type not in (None, "text") or not isinstance(text, str):
        return ""
    return text


def _message_content_text(message: Any) -> str:
    """Return text from both raw and Gradio-normalized chatbot messages."""
    return _content_text(_message_field(message, "content")).strip()


def _is_pending_assistant_message(message: Any) -> bool:
    if not message or _message_field(message, "role") != "assistant":
        return False
    content = _message_content_text(message)
    has_default_status = DEFAULT_PENDING_STATUS in content
    has_tool_status = "현재 " in content and " 실행 중" in content
    return content.startswith("...") and (has_default_status or has_tool_status)


def _replace_pending_status(history: list[dict[str, Any]], status_text: str) -> list[dict[str, Any]]:
    pending_message = _pending_assistant_message(status_text)
    if history and _is_pending_assistant_message(history[-1]):
        return [*history[:-1], pending_message]
    return [*history, pending_message]


def _replace_pending_with_answer(history: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    assistant_message = {"role": "assistant", "content": answer}
    if history and _is_pending_assistant_message(history[-1]):
        return [*history[:-1], assistant_message]
    return [*history, assistant_message]


def _saved_chatbot_history(conversation_id: str) -> list[dict[str, str]]:
    return [
        message
        for message in runtime.load_messages_for_chatbot(conversation_id)
        if not _is_pending_assistant_message(message)
    ]


def _conversation_rows() -> list[dict[str, str]]:
    rows = runtime.app_store.list_conversations()
    return [
        {
            "conversation_id": row["conversation_id"],
            "title": row["title"] or "새 대화",
            "preview": (row.get("last_message") or "").replace("\n", " ")[:54],
        }
        for row in rows
    ]


def _conversation_selector_update(selected_id: str | None = None) -> Any:
    choices = [(row["title"], row["conversation_id"]) for row in _conversation_rows()]
    return gr.update(choices=choices, value=selected_id or None)


def queue_user_message(
    message: str,
    history: list[dict[str, Any]] | None,
    conversation_id: str | None,
) -> tuple:
    history = history or []
    message = (message or "").strip()
    if not message:
        return (
            history,
            {},
            conversation_id or "",
            gr.update(value="", interactive=True),
            "",
            gr.update(interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    active_conversation_id = runtime.ensure_conversation(conversation_id or None, message)
    history = [
        *history,
        {"role": "user", "content": message},
        _pending_assistant_message(),
    ]
    return (
        history,
        {"mode": "pending"},
        active_conversation_id,
        gr.update(value="", interactive=False),
        message,
        gr.update(interactive=False),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        _conversation_selector_update(active_conversation_id),
    )


def finish_agent_response(
    pending_message: str,
    history: list[dict[str, Any]] | None,
    conversation_id: str | None,
    secret_mode: bool = False,
) -> Any:
    history = history or []
    pending_message = (pending_message or "").strip()
    if not pending_message:
        yield (
            history,
            {},
            conversation_id or "",
            gr.update(interactive=True),
            "",
            gr.update(interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
        return

    active_conversation_id = conversation_id or None
    for event in runtime.stream_agent(pending_message, active_conversation_id, secret_mode=secret_mode): # 런타임 호출에 secret_mode 전달
        if event.status_text:
            history = _replace_pending_status(history, event.status_text)
            yield (
                history,
                {"mode": "pending", "status": event.status_text},
                conversation_id or "",
                gr.update(interactive=False),
                pending_message,
                gr.update(interactive=False),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        if event.result:
            history = _replace_pending_with_answer(history, event.result.answer)
            yield (
                history,
                event.result.trace,
                event.result.conversation_id,
                gr.update(interactive=True),
                "",
                gr.update(interactive=True),
                *_saved_item_tables(),
                _conversation_selector_update(event.result.conversation_id),
            )
            return


def new_chat() -> tuple:
    return (_chat_notice(), {}, "", *_saved_item_tables(), _conversation_selector_update(None))


def load_chat(conversation_id: str | None) -> tuple:
    if not conversation_id:
        return (_chat_notice(), "", *_saved_item_tables())
    return (
        _saved_chatbot_history(conversation_id),
        conversation_id,
        *_saved_item_tables(),
    )


def archive_chat(conversation_id: str | None) -> tuple:
    runtime.archive_conversation(conversation_id)
    return (_chat_notice(), {}, "", *_saved_item_tables(), _conversation_selector_update(None))


def delete_chat(conversation_id: str | None) -> tuple:
    if conversation_id:
        runtime.delete_conversation(conversation_id)
    return (_chat_notice(), {}, "", *_saved_item_tables(), _conversation_selector_update(None))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Kanana Schedule Agent") as demo:
        conversation_id = gr.Textbox(value="", visible=False, elem_id="selected-conversation-id", container=False)
        pending_message = gr.State("")
        gr.HTML(
            f"""
            <div class="kanana-topbar">
              <button
                id="conversation-drawer-toggle"
                type="button"
                aria-label="대화 내역 펼치기"
                aria-expanded="false"
                title="대화 내역 펼치기"
              ><span aria-hidden="true">☰</span></button>
              <div class="brand-lockup">
                <span>Smart Schedule Agent</span>
              </div>
            </div>
            """
        )
        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("채팅"):
                with gr.Row(elem_id="kanana-shell", elem_classes=["conversation-drawer-collapsed"]):
                    # 대화 전용 drawer
                    with gr.Column(
                        scale=0,
                        min_width=200,
                        elem_id="conversation-drawer",
                        elem_classes=["conversation-drawer"],
                    ):
                        new_btn = gr.Button(
                            "새 대화",
                            elem_classes=["primary-action"],
                        )

                        gr.HTML(
                            "<div class='conversation-list-title'>대화 내역</div>",
                            container=False,
                        )

                        conversation_selector = gr.Radio(
                            choices=[
                                (row["title"], row["conversation_id"])
                                for row in _conversation_rows()
                            ],
                            value=None,
                            show_label=False,
                            elem_id="conversation-selector",
                        )

                        archive_btn = gr.Button(
                            "현재 대화 보관",
                            elem_classes=["ghost-action"],
                        )
                        delete_btn = gr.Button(
                            "저장된 대화 삭제",
                            elem_classes=["danger-action"],
                        )

                        gr.HTML(
                            """
                            <div
                                id="conversation-drawer-resizer"
                                role="separator"
                                aria-orientation="vertical"
                            ></div>
                            """,
                            container=False,
                        )

                    # 일정 전용 패널
                    with gr.Column(scale=2, min_width=0, elem_classes=["sidebar", "schedule-sidebar"]):
                        initial_schedules, initial_reminders, initial_todos = (_saved_item_tables())

                        gr.HTML(
                            "<div class='conversation-list-title'>내 일정</div>",
                            container=False,
                        )

                        with gr.Column(
                            min_width=0,
                            elem_id="saved-items-fields",
                            elem_classes=["saved-items-fields"],
                        ):
                            with gr.Column(
                                min_width=0,
                                elem_id="schedule-field",
                                elem_classes=["saved-item-field"],
                            ):
                                saved_schedules = gr.Dataframe(
                                    value=initial_schedules,
                                    headers=["날짜", "시작", "종료", "제목", "참석자"],
                                    datatype=["str", "str", "str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    show_label=True,
                                    label="일정",
                                    elem_id="schedule-table",
                                    show_search="filter"
                                )

                            gr.HTML(
                                """
                                <div
                                    class="saved-field-resizer"
                                    data-resize-before="schedule-field"
                                    data-resize-after="reminder-field"
                                    role="separator"
                                    aria-orientation="horizontal"
                                    title="드래그해서 높이 조절"
                                ></div>
                                """,
                                container=False,
                            )

                            with gr.Column(
                                min_width=0,
                                elem_id="reminder-field",
                                elem_classes=["saved-item-field"],
                            ):
                                saved_reminders = gr.Dataframe(
                                    value=initial_reminders,
                                    headers=["날짜", "시간", "제목"],
                                    datatype=["str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    show_label=True,
                                    label="리마인더",
                                    elem_id="reminder-table",
                                    show_search="filter"
                                )

                            gr.HTML(
                                """
                                <div
                                    class="saved-field-resizer"
                                    data-resize-before="reminder-field"
                                    data-resize-after="todo-field"
                                    role="separator"
                                    aria-orientation="horizontal"
                                    title="드래그해서 높이 조절"
                                ></div>
                                """,
                                container=False,
                            )

                            with gr.Column(
                                min_width=0,
                                elem_id="todo-field",
                                elem_classes=["saved-item-field"],
                            ):
                                saved_todos = gr.Dataframe(
                                    value=initial_todos,
                                    headers=["우선순위", "기한", "시간", "제목"],
                                    datatype=["str", "str", "str", "str"],
                                    interactive=False,
                                    wrap=True,
                                    show_label=True,
                                    label="할 일",
                                    elem_id="todo-table",
                                    show_search="filter"
                                )
                        gr.HTML(
                            '<div id="schedule-panel-resizer" role="separator" aria-orientation="vertical" title="드래그해서 일정 패널 너비 조절"></div>',
                            container=False,
                        )
                    # 채팅 패널
                    with gr.Column(scale=4, min_width=0, elem_classes=["chat-panel"]):
                        chatbot = gr.Chatbot(
                            value=_chat_notice(),
                            height=680,
                            show_label=False,
                            elem_id="kanana-chatbot",
                            placeholder="",
                        )
                        with gr.Row(elem_classes=["composer"]):
                            textbox = gr.Textbox(
                                placeholder="",
                                show_label=False,
                                lines=2,
                                elem_id="kanana-input",
                            )
                            with gr.Column(min_width=90, scale=0, elem_classes=["send-controls"]):
                                secret_mode = gr.Checkbox(label="Secret", value=False, elem_id="secret-mode") # 시크릿 대화 체크박스 컴포넌트
                                send_btn = gr.Button("↑", elem_id="kanana-send", elem_classes=["send-button"])
            with gr.Tab("상세"):
                with gr.Row(elem_classes=["details-layout"]):
                    with gr.Column(scale=1, min_width=720, elem_classes=["detail-card", "trace-detail-card"]):
                        gr.HTML("<div class='trace-title'>마지막 에이전트 실행 Trace</div>")
                        trace_json = gr.JSON(
                            label="trace 페이로드",
                            value={},
                            elem_id="trace-json",
                            open=True,
                            min_height=620,
                            max_height=780,
                        )
        send_outputs = [
            chatbot,
            trace_json,
            conversation_id,
            textbox,
            pending_message,
            send_btn,
            saved_schedules,
            saved_reminders,
            saved_todos,
            conversation_selector,
        ]
        finish_outputs = [
            chatbot,
            trace_json,
            conversation_id,
            textbox,
            pending_message,
            send_btn,
            saved_schedules,
            saved_reminders,
            saved_todos,
            conversation_selector,
        ]
        send_btn.click(
            queue_user_message,
            inputs=[textbox, chatbot, conversation_id],
            outputs=send_outputs,
            queue=False,
        ).then(
            finish_agent_response,
            inputs=[pending_message, chatbot, conversation_id, secret_mode],
            outputs=finish_outputs,
            show_progress="hidden",
        )
        new_btn.click(new_chat, outputs=[chatbot, trace_json, conversation_id, saved_schedules, saved_reminders, saved_todos, conversation_selector])
        archive_btn.click(
            archive_chat,
            inputs=[conversation_id],
            outputs=[chatbot, trace_json, conversation_id, saved_schedules, saved_reminders, saved_todos, conversation_selector],
        )
        delete_btn.click(
            delete_chat,
            inputs=[conversation_id],
            outputs=[chatbot, trace_json, conversation_id, saved_schedules, saved_reminders, saved_todos, conversation_selector],
            js=DELETE_CONVERSATION_CONFIRM_JS,
            queue=False,
        )
        conversation_selector.input(
            load_chat,
            inputs=[conversation_selector],
            outputs=[
                chatbot,
                conversation_id,
                saved_schedules,
                saved_reminders,
                saved_todos,
            ],
            show_progress="hidden",
        )
    return demo


if __name__ == "__main__":
    if not CONFIG.has_openai_key:
        print("주의: 프롬프트 기반 에이전트 채팅에는 .env의 PROXY_TOKEN이 필요합니다.")
    build_demo().launch(css_paths=[str(CSS_PATH)], head=ENTER_TO_SEND_HEAD)
