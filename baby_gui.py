#!/usr/bin/env python3
"""BabyX desktop interface: camera awareness, teaching, chat, and feedback."""

import os
import sys
import threading
import time
from tkinter import messagebox, simpledialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps

from baby_brain import generate_reply, get_last_context
from memory import BabyMemory
from object_store import remember_object
from rl_agent import RLAgent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SPRITES = {
    "idle": "idle.png",
    "center": "idle.png",
    "left": "look_left.png",
    "right": "look_right.png",
    "thinking": "thinking.png",
    "happy": "happy.png",
}

COLOR = {
    "page": "#0D1112",
    "surface": "#151C1D",
    "surface_alt": "#1B2525",
    "surface_soft": "#202C2A",
    "border": "#2D3A38",
    "text": "#F7F1E7",
    "muted": "#ABB8B2",
    "subtle": "#788A84",
    "accent": "#E9935E",
    "accent_hover": "#D77E4B",
    "accent_text": "#231711",
    "success": "#A7D6B2",
    "user_bubble": "#31433E",
    "baby_bubble": "#222D2C",
}


def font(size, weight="normal"):
    return ctk.CTkFont(family="Helvetica Neue", size=size, weight=weight)


def avatar(path, size=36):
    try:
        with Image.open(path) as source:
            image = ImageOps.fit(
                source.convert("RGBA"), (size, size),
                method=Image.Resampling.LANCZOS,
            )
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        image.putalpha(mask)
        return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))
    except (OSError, ValueError):
        return None


def sprite(path, bounds=(160, 224)):
    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
        image.thumbnail(bounds, Image.Resampling.LANCZOS)
        return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
    except (OSError, ValueError):
        return None


class BabyXApp:
    def __init__(self, root, preview=False):
        self.root = root
        self.preview = preview
        self.root.title("BabyX — Vision companion")
        self.root.geometry("1200x800")
        self.root.minsize(960, 650)
        self.root.configure(fg_color=COLOR["page"])
        ctk.set_appearance_mode("dark")

        self.memory = BabyMemory()
        self.rl = RLAgent()
        self.vision = None
        self.camera_error = None
        self.current_vision_state = None
        self._vision_thread_running = False
        self._closing = False
        self._current_mood = None
        self.baby_images = {}
        self.avatar_baby = avatar(os.path.join(ASSETS_DIR, "happy.png"))
        self.avatar_user = avatar(os.path.join(ASSETS_DIR, "me_dp.png"))

        self._build_layout()
        self._load_baby_images()

        if preview:
            self.current_vision_state = {
                "focus_type": "object", "focus_label": "cup",
                "focus_side": "center", "confidence": 0.94,
            }
            self.teach_button.configure(state="disabled")
            self.retry_button.configure(state="disabled")
            self._add_message("baby", "Hi, I'm BabyX. Show me something or ask what I can see.")
            self._add_message("me", "What are you looking at?")
            self._add_message("baby", "I can see a cup. You can teach me what you call it.")
        else:
            # Paint the window first. Camera/model startup may take a moment.
            self.root.after(120, self._start_camera)

        self._schedule_gui_refresh()

    def _surface(self, parent, radius=20, color="surface"):
        return ctk.CTkFrame(
            parent, fg_color=COLOR[color], corner_radius=radius,
            border_width=1, border_color=COLOR["border"],
        )

    def _build_layout(self):
        page = ctk.CTkFrame(self.root, fg_color="transparent")
        page.pack(fill="both", expand=True, padx=24, pady=20)

        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))
        header_left = ctk.CTkFrame(header, fg_color="transparent")
        header_left.pack(side="left")
        ctk.CTkLabel(
            header_left, text="BABYX", font=font(29, "bold"),
            text_color=COLOR["text"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            header_left, text="A CURIOUS COMPANION", font=font(11, "bold"),
            text_color=COLOR["accent"], anchor="w",
        ).pack(anchor="w", pady=(0, 2))

        status_shell = ctk.CTkFrame(
            header, fg_color=COLOR["surface_alt"], corner_radius=13,
            border_width=1, border_color=COLOR["border"],
        )
        status_shell.pack(side="right", pady=8)
        self.camera_status = ctk.CTkLabel(
            status_shell, text="Starting camera", font=font(12, "bold"),
            text_color=COLOR["muted"],
        )
        self.camera_status.pack(padx=16, pady=9)

        ctk.CTkFrame(page, height=1, fg_color=COLOR["border"], corner_radius=0).pack(fill="x")

        content = ctk.CTkFrame(page, fg_color="transparent")
        content.pack(fill="both", expand=True, pady=(18, 0))
        content.columnconfigure(0, minsize=308, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(
            content, width=306, fg_color="transparent", corner_radius=0,
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        portrait = self._surface(sidebar)
        portrait.pack(fill="x", pady=(0, 12))
        portrait_head = ctk.CTkFrame(portrait, fg_color="transparent")
        portrait_head.pack(fill="x", padx=18, pady=(15, 4))
        ctk.CTkLabel(
            portrait_head, text="THE COMPANION", font=font(11, "bold"),
            text_color=COLOR["muted"],
        ).pack(side="left")
        self.mood_label = ctk.CTkLabel(
            portrait_head, text="IDLE", font=font(11, "bold"),
            text_color=COLOR["accent"],
        )
        self.mood_label.pack(side="right")
        self.baby_label = ctk.CTkLabel(portrait, text="")
        self.baby_label.pack(pady=(2, 12))

        focus_card = self._surface(sidebar)
        focus_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            focus_card, text="CURRENT FOCUS", font=font(11, "bold"),
            text_color=COLOR["accent"], anchor="w",
        ).pack(fill="x", padx=18, pady=(16, 2))
        self.focus_title = ctk.CTkLabel(
            focus_card, text="Waiting to see", font=font(20, "bold"),
            text_color=COLOR["text"], anchor="w", wraplength=265,
        )
        self.focus_title.pack(fill="x", padx=18)
        self.memory_text = ctk.CTkLabel(
            focus_card, text="Show an object to the camera.",
            font=font(12), text_color=COLOR["muted"],
            anchor="w", justify="left", wraplength=265,
        )
        self.memory_text.pack(fill="x", padx=18, pady=(5, 15))
        self.teach_button = ctk.CTkButton(
            focus_card, text="Teach this object", height=40,
            fg_color=COLOR["accent"], hover_color=COLOR["accent_hover"],
            text_color=COLOR["accent_text"], font=font(13, "bold"),
            corner_radius=10, command=self._on_teach_clicked,
        )
        self.teach_button.pack(fill="x", padx=18, pady=(0, 10))
        self.retry_button = ctk.CTkButton(
            focus_card, text="Reconnect camera", height=32,
            fg_color="transparent", hover_color=COLOR["surface_soft"],
            border_width=1, border_color=COLOR["border"],
            text_color=COLOR["muted"], font=font(12),
            corner_radius=9, command=self._start_camera,
        )
        self.retry_button.pack(fill="x", padx=18, pady=(0, 17))

        guide = self._surface(sidebar, radius=18, color="surface_alt")
        guide.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            guide, text="HOW IT WORKS", font=font(11, "bold"),
            text_color=COLOR["muted"], anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 5))
        ctk.CTkLabel(
            guide,
            text="Show an object. Give it a name. Then ask BabyX what it sees.",
            font=font(12), text_color=COLOR["text"],
            wraplength=265, justify="left", anchor="w",
        ).pack(fill="x", padx=18, pady=(0, 16))

        self.chat_panel = self._surface(content, radius=22)
        self.chat_panel.grid(row=0, column=1, sticky="nsew")
        self.chat_panel.columnconfigure(0, weight=1)
        self.chat_panel.rowconfigure(2, weight=1)

        chat_header = ctk.CTkFrame(self.chat_panel, fg_color="transparent")
        chat_header.grid(row=0, column=0, sticky="ew", padx=24, pady=(21, 17))
        ctk.CTkLabel(
            chat_header, text="Conversation", font=font(22, "bold"),
            text_color=COLOR["text"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            chat_header, text="Ask a question, or teach BabyX something new.",
            font=font(12), text_color=COLOR["muted"], anchor="w",
        ).pack(anchor="w", pady=(2, 0))
        ctk.CTkFrame(
            self.chat_panel, height=1, fg_color=COLOR["border"], corner_radius=0,
        ).grid(row=1, column=0, sticky="ew")

        self.chat_area = ctk.CTkScrollableFrame(
            self.chat_panel, fg_color="transparent", corner_radius=0,
        )
        self.chat_area.grid(row=2, column=0, sticky="nsew", padx=18, pady=(10, 5))
        self.empty_state = ctk.CTkFrame(
            self.chat_area, fg_color=COLOR["surface_alt"],
            corner_radius=16, border_width=1, border_color=COLOR["border"],
        )
        self.empty_state.pack(padx=24, pady=58)
        ctk.CTkLabel(
            self.empty_state, text="Start a conversation", font=font(19, "bold"),
            text_color=COLOR["text"],
        ).pack(padx=38, pady=(27, 6))
        ctk.CTkLabel(
            self.empty_state,
            text="BabyX is ready to observe, answer, and learn with you.",
            font=font(13), text_color=COLOR["muted"], wraplength=340,
        ).pack(padx=38, pady=(0, 27))

        composer = ctk.CTkFrame(self.chat_panel, fg_color="transparent")
        composer.grid(row=3, column=0, sticky="ew", padx=22, pady=(3, 19))
        composer.columnconfigure(0, weight=1)
        quick = ctk.CTkFrame(composer, fg_color="transparent")
        quick.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 11))
        ctk.CTkLabel(
            quick, text="TRY ASKING", font=font(10, "bold"),
            text_color=COLOR["subtle"],
        ).pack(side="left", padx=(0, 10))
        for label, prompt in (
            ("What do you see?", "what are you looking at?"),
            ("Do you know this?", "do you know this?"),
            ("Who am I?", "who am I?"),
        ):
            ctk.CTkButton(
                quick, text=label, width=0, height=28,
                fg_color=COLOR["surface_alt"], hover_color=COLOR["surface_soft"],
                border_width=1, border_color=COLOR["border"],
                text_color=COLOR["muted"], font=font(11), corner_radius=9,
                command=lambda value=prompt: self._set_quick_text(value),
            ).pack(side="left", padx=(0, 7))

        self.entry = ctk.CTkEntry(
            composer, placeholder_text="Ask BabyX anything...", height=47,
            fg_color=COLOR["surface_alt"], border_color=COLOR["border"],
            border_width=1, text_color=COLOR["text"],
            placeholder_text_color=COLOR["subtle"],
            font=font(14), corner_radius=11,
        )
        self.entry.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda _event: self._on_send_clicked())
        self.send_button = ctk.CTkButton(
            composer, text="Send", width=92, height=47,
            fg_color=COLOR["accent"], hover_color=COLOR["accent_hover"],
            text_color=COLOR["accent_text"], font=font(13, "bold"),
            corner_radius=11, command=self._on_send_clicked,
        )
        self.send_button.grid(row=1, column=1)

    def _load_baby_images(self):
        for mood, filename in SPRITES.items():
            self.baby_images[mood] = sprite(os.path.join(ASSETS_DIR, filename))
        self._set_baby_image("idle")

    def _set_baby_image(self, mood):
        if mood == self._current_mood:
            return
        image = self.baby_images.get(mood)
        if image is not None:
            self.baby_label.configure(image=image)
        self.mood_label.configure(text=mood.upper())
        self._current_mood = mood

    def _add_message(self, who, text, feedback_state="chat"):
        if self.empty_state is not None:
            self.empty_state.destroy()
            self.empty_state = None

        is_baby = who == "baby"
        outer = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        outer.pack(fill="x", pady=(5, 9))
        row = ctk.CTkFrame(outer, fg_color="transparent")
        row.pack(anchor="w" if is_baby else "e", padx=6)

        avatar_image = self.avatar_baby if is_baby else self.avatar_user
        if avatar_image is not None:
            avatar_label = ctk.CTkLabel(row, text="", image=avatar_image)
            avatar_label.pack(side="left" if is_baby else "right", anchor="n", padx=(0, 10) if is_baby else (10, 0), pady=15)

        content = ctk.CTkFrame(row, fg_color="transparent")
        content.pack(side="left" if is_baby else "right")
        ctk.CTkLabel(
            content, text="BABYX" if is_baby else "YOU", font=font(10, "bold"),
            text_color=COLOR["accent"] if is_baby else COLOR["subtle"],
        ).pack(anchor="w" if is_baby else "e", padx=3, pady=(0, 4))
        bubble = ctk.CTkFrame(
            content,
            fg_color=COLOR["baby_bubble"] if is_baby else COLOR["user_bubble"],
            corner_radius=15,
        )
        bubble.pack(anchor="w" if is_baby else "e")
        ctk.CTkLabel(
            bubble, text=text, font=font(14), text_color=COLOR["text"],
            wraplength=440, justify="left", anchor="w",
        ).pack(padx=15, pady=12)

        if is_baby:
            feedback = ctk.CTkFrame(content, fg_color="transparent")
            feedback.pack(anchor="w", pady=(7, 0))
            button_style = dict(
                height=27, fg_color="transparent", hover_color=COLOR["surface_soft"],
                border_width=1, border_color=COLOR["border"],
                text_color=COLOR["muted"], font=font(11), corner_radius=8,
            )
            good = ctk.CTkButton(feedback, text="Helpful", width=76, **button_style)
            bad = ctk.CTkButton(feedback, text="Not helpful", width=92, **button_style)
            good.pack(side="left", padx=(0, 6))
            bad.pack(side="left")
            good.configure(command=lambda: self._thumb_feedback(True, feedback_state, text, good, bad))
            bad.configure(command=lambda: self._thumb_feedback(False, feedback_state, text, good, bad))

        self._scroll_chat_to_bottom()

    def _thumb_feedback(self, is_good, state_key, reply, good_button, bad_button):
        if self.preview:
            return
        try:
            # Each message rates its own reply, not whichever reply came last.
            self.rl.record_outcome(
                state_key=state_key, action=reply,
                reward=1.0 if is_good else -1.0,
            )
            selected = good_button if is_good else bad_button
            selected.configure(
                text="Feedback saved", fg_color=COLOR["surface_soft"],
                text_color=COLOR["success"],
            )
            good_button.configure(state="disabled")
            bad_button.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Feedback", f"Could not save feedback: {exc}")

    def _scroll_chat_to_bottom(self):
        try:
            canvas = getattr(self.chat_area, "_parent_canvas", None)
            if canvas is not None:
                self.root.after_idle(lambda: canvas.yview_moveto(1.0))
        except Exception:
            pass

    def _set_quick_text(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus()

    def _on_send_clicked(self, preset_msg=None):
        user_msg = (preset_msg if preset_msg is not None else self.entry.get()).strip()
        if not user_msg:
            return
        if preset_msg is None:
            self.entry.delete(0, "end")
        self._add_message("me", user_msg)
        reply = generate_reply(user_msg, self.current_vision_state, self.memory, self.rl)
        state_key, _reply = get_last_context()
        self._add_message("baby", reply, feedback_state=state_key or "chat")

    def _on_teach_clicked(self):
        if self.preview:
            return
        state = self.current_vision_state
        if not state or state.get("focus_type") != "object":
            messagebox.showinfo("Teach an object", "Show an object to the camera first, then try again.")
            return

        base_label = state.get("focus_label") or "this object"
        crop = state.get("current_crop")
        confidence = state.get("confidence", 0.0)
        taught_name = simpledialog.askstring(
            "Teach an object",
            f"What do you call this?\nDetected as {base_label} ({confidence:.0%} confidence)",
            parent=self.root,
        )
        if not taught_name or not taught_name.strip():
            return
        taught_name = taught_name.strip()

        self.memory.teach(base_label, taught_name)
        if crop is not None:
            try:
                remember_object(base_label, taught_name, crop)
            except Exception as exc:
                print(f"Could not save visual object: {exc}")
        try:
            self.rl.record_outcome(
                state_key=f"teaching|{base_label}",
                action=f"remembered {taught_name}", reward=1,
            )
        except Exception as exc:
            print(f"Could not save teaching feedback: {exc}")

        self._add_message("baby", f"Got it. I'll remember this as {taught_name}.", feedback_state=f"teaching|{base_label}")

    def _start_camera(self):
        if self.preview or self._closing or self._vision_thread_running:
            return
        self.camera_error = None
        self.camera_status.configure(text="Starting camera", text_color=COLOR["muted"])
        self._vision_thread_running = True
        threading.Thread(target=self._vision_loop, daemon=True).start()

    def _vision_loop(self):
        vision = None
        try:
            # Vision imports and model/camera initialization can be slow.
            # Keep them off Tk's event loop so the interface remains responsive.
            from baby_vision import BabyVision

            vision = BabyVision()
            if self._closing or not self._vision_thread_running:
                return
            self.vision = vision
            while self._vision_thread_running and not self._closing:
                state, _frame = vision.step()
                self.current_vision_state = state
                time.sleep(0.05)
        except Exception as exc:
            self.camera_error = str(exc)
        finally:
            if vision is not None:
                vision.release()
            self.vision = None
            self._vision_thread_running = False

    def _decide_baby_image_key(self, state):
        if not state:
            return "idle"
        focus_type = state.get("focus_type")
        if focus_type == "object":
            return "happy" if self.memory.recall(state.get("focus_label")) else "thinking"
        if focus_type == "face":
            side = state.get("focus_side")
            return side if side in ("left", "right") else "center"
        return "idle"

    def _schedule_gui_refresh(self):
        if self._closing:
            return
        state = self.current_vision_state
        self._set_baby_image(self._decide_baby_image_key(state))

        if self.preview:
            self.camera_status.configure(text="Preview mode", text_color=COLOR["accent"])
        elif self.vision is None and self.camera_error:
            self.camera_status.configure(text="Camera unavailable", text_color=COLOR["accent"])
        elif self.vision is not None:
            self.camera_status.configure(text="Camera connected", text_color=COLOR["success"])

        self.retry_button.configure(state="normal" if not self.preview and not self._vision_thread_running else "disabled")
        can_teach = not self.preview and bool(state) and state.get("focus_type") == "object"
        self.teach_button.configure(
            state="normal" if can_teach else "disabled",
            fg_color=COLOR["accent"] if can_teach else COLOR["surface_soft"],
            text_color=COLOR["accent_text"] if can_teach else COLOR["subtle"],
        )
        if not state:
            self.focus_title.configure(text="Waiting to see")
            detail = self.camera_error or "Show an object to the camera."
            self.memory_text.configure(text=detail)
        elif state.get("focus_type") == "object":
            raw_label = state.get("focus_label") or "Object"
            taught = self.memory.recall(raw_label)
            self.focus_title.configure(text=taught or raw_label.title())
            confidence = state.get("confidence", 0.0)
            detail = f"Recognized object · {confidence:.0%} confidence" if taught else f"New object · {confidence:.0%} confidence"
            self.memory_text.configure(text=detail)
        elif state.get("focus_type") == "face":
            self.focus_title.configure(text="A face in view")
            side = state.get("focus_side") or "center"
            self.memory_text.configure(text=f"Looking {side} · ready to chat")
        else:
            self.focus_title.configure(text="Looking around")
            self.memory_text.configure(text="Hold up an object or ask me a question.")

        self.root.after(150, self._schedule_gui_refresh)

    def shutdown(self):
        self._closing = True
        self._vision_thread_running = False
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = BabyXApp(root, preview="--preview" in sys.argv)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()
