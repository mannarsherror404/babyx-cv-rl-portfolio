#!/usr/bin/env python3
# ============================================================
# BabyX GUI — stable macOS version + RL hook + feedback buttons + crop teach
# - Left column: Baby sprite + Memory box + Info box
# - Right column: Chat + sticky bottom input bar + quick buttons
# - Mood follows vision (idle / thinking / happy / look_left / look_right)
# - 👍 / 👎 under each Baby bubble to reward style
# - Teach now: grabs vision crop, saves it, and logs features
# ============================================================

import os
import time
import threading
import customtkinter as ctk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageDraw

from baby_vision import BabyVision
from memory import BabyMemory
from baby_brain import generate_reply, reward_last_reply, set_last_context
from rl_agent import RLAgent as RLBaby  # RL policy
from object_store import remember_object, identify_object, stats  # optional: clear_store

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

WINDOW_BG_IMAGE = os.path.join(ASSETS_DIR, "glass_bg.png")
AVATAR_BOT_PATH = os.path.join(ASSETS_DIR, "happy.png")
AVATAR_USER_PATH = os.path.join(ASSETS_DIR, "me_dp.png")
AVATAR_SIZE = (40, 40)

BABY_DISPLAY_SIZE = (300, 300)

BABY_IMAGE_PATHS = {
    "idle":      os.path.join(ASSETS_DIR, "idle.png"),
    "left":      os.path.join(ASSETS_DIR, "look_left.png"),
    "right":     os.path.join(ASSETS_DIR, "look_right.png"),
    "center":    os.path.join(ASSETS_DIR, "idle.png"),
    "thinking":  os.path.join(ASSETS_DIR, "thinking.png"),
    "happy":     os.path.join(ASSETS_DIR, "happy.png"),
    "unknown":   os.path.join(ASSETS_DIR, "thinking.png"),
}

# ---------- COLORS / STYLE ----------
APPEARANCE_MODE = "dark"
THEME_COLOR = "blue"

CARD_BG_COLOR = "#2B2B2B"
CARD_BORDER_COLOR = "#555555"
CHAT_PANEL_COLOR = "#1E1E1E"
CHAT_SCROLL_BG = "#000000"
INPUT_BAR_COLOR = "#222222"

BOT_BUBBLE_COLOR = "#2E2E2E"
USER_BUBBLE_COLOR = "#284A7A"
BOT_TEXT_COLOR = "#E5FFCC"
USER_TEXT_COLOR = "#FFFFFF"

CARD_RADIUS = 20
BUBBLE_RADIUS = 20


# ---------- IMAGE HELPERS ----------
def make_round_image(path, size):
    """circular avatar (for chat bubbles)"""
    try:
        img = Image.open(path).convert("RGBA").resize(size)
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)
        img.putalpha(mask)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception as e:
        print(f"⚠️ Avatar load error for {path}: {e}")
        return None


def make_square_image(path, size):
    """square sprite (for big baby on the left)"""
    try:
        img = Image.open(path).convert("RGBA").resize(size)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception as e:
        print(f"⚠️ Baby sprite load error for {path}: {e}")
        return None


# ---------- MAIN APP ----------
class BabyXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BabyX")

        # theme first
        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(THEME_COLOR)

        # background gradient
        if os.path.exists(WINDOW_BG_IMAGE):
            try:
                from PIL import Image  # local import in case PIL is missing
                bg_img = Image.open(WINDOW_BG_IMAGE)
                self.bg_photo = ctk.CTkImage(
                    light_image=bg_img,
                    dark_image=bg_img,
                    size=(1920, 1080)
                )
                bg_label = ctk.CTkLabel(self.root, text="", image=self.bg_photo)
                bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
            except Exception as e:
                print(f"⚠️ Could not load background image: {e}")

        # core runtime
        self.vision = BabyVision()        # webcam + detection
        self.memory = BabyMemory()        # taught names + (now) visual features
        self.rl = RLBaby()                # RL policy store / scoring
        self.current_vision_state = None  # will be updated by vision thread

        # UI state
        self.baby_images = {}
        self.avatar_bot = None
        self.avatar_user = None
        self.last_baby_reply = None  # last baby message for 👍/👎

        # build UI
        self._build_layout()
        self._load_baby_images()

        # start vision polling thread
        self._vision_thread_running = True
        threading.Thread(target=self._vision_loop, daemon=True).start()

        # schedule refresh loop
        self._schedule_gui_refresh()

    # ------------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------------
    def _build_layout(self):
        # MAIN WRAPPER
        self.main = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main.pack(fill="both", expand=True, padx=10, pady=10)

        self.main.columnconfigure(0, weight=0)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        # LEFT COLUMN
        self.left = ctk.CTkFrame(self.main, fg_color="transparent")
        self.left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        # rows: 0 baby sprite, 1 memory box, 2 info box
        self.left.rowconfigure(0, weight=1)
        self.left.rowconfigure(1, weight=0)
        self.left.rowconfigure(2, weight=0)

        # Baby sprite at top
        self.baby_label = ctk.CTkLabel(self.left, text="")
        self.baby_label.grid(row=0, column=0, pady=(40, 10), sticky="n")

        # MEMORY BOX (live vision + recall info)
        self.memory_frame = ctk.CTkFrame(
            self.left,
            fg_color=CARD_BG_COLOR,
            border_color=CARD_BORDER_COLOR,
            border_width=1,
            corner_radius=CARD_RADIUS,
        )
        self.memory_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0, 10)
        )

        ctk.CTkLabel(
            self.memory_frame,
            text="🧠 Memory",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFFFFF",
            anchor="w",
            justify="left"
        ).pack(anchor="w", padx=10, pady=(8, 2))

        self.memory_text = ctk.CTkLabel(
            self.memory_frame,
            text="No recent detections.",
            justify="left",
            text_color="#CCCCCC",
            anchor="w",
            wraplength=260
        )
        self.memory_text.pack(anchor="w", padx=10, pady=(0, 10))

        # INFO BOX (how to talk / teach)
        self.info_box = ctk.CTkFrame(
            self.left,
            fg_color=CARD_BG_COLOR,
            border_color=CARD_BORDER_COLOR,
            border_width=1,
            corner_radius=CARD_RADIUS,
        )
        self.info_box.grid(
            row=2,
            column=0,
            sticky="sew",
            padx=10,
            pady=(0, 10)
        )

        ctk.CTkLabel(
            self.info_box,
            text="🍼 BabyX",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFFFFF",
            anchor="w",
            justify="left"
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            self.info_box,
            text=(
                "I see faces, hands, and what you're holding.\n\n"
                "📸 Teach:\n"
                " • Hold an object in your hand\n"
                " • Press “Teach”\n"
                " • Tell me what it is (like “my phone”)\n\n"
                "Ask me:\n"
                " • what are you looking at?\n"
                " • do you know this?\n"
                " • who am I?\n"
            ),
            justify="left",
            text_color="#DDDDDD",
            anchor="w"
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # RIGHT COLUMN (chat panel)
        self.right = ctk.CTkFrame(
            self.main,
            fg_color=CHAT_PANEL_COLOR,
            border_color=CARD_BORDER_COLOR,
            border_width=1,
            corner_radius=25,
        )
        self.right.grid(row=0, column=1, sticky="nsew")
        self.main.rowconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=1)

        # rows in right panel:
        # 0 chat scroll
        # 1 input bar
        self.right.rowconfigure(0, weight=1)
        self.right.rowconfigure(1, weight=0)
        self.right.columnconfigure(0, weight=1)

        # scrollable chat frame
        self.chat_area = ctk.CTkScrollableFrame(
            self.right,
            fg_color=CHAT_SCROLL_BG,
            corner_radius=20
        )
        self.chat_area.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(10, 5)
        )

        # avatars for chat bubbles
        self.avatar_bot = make_round_image(AVATAR_BOT_PATH, AVATAR_SIZE)
        self.avatar_user = make_round_image(AVATAR_USER_PATH, AVATAR_SIZE)

        # bottom input row
        self.input_bar = ctk.CTkFrame(
            self.right,
            fg_color=INPUT_BAR_COLOR,
            corner_radius=15,
        )
        self.input_bar.grid(
            row=1,
            column=0,
            sticky="sew",
            padx=10,
            pady=(0, 10)
        )
        self.input_bar.columnconfigure(0, weight=1)

        # entry box
        self.entry = ctk.CTkEntry(
            self.input_bar,
            placeholder_text="Type a message...",
            corner_radius=15
        )
        self.entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=5,
            pady=5
        )

        # teach button
        self.teach_button = ctk.CTkButton(
            self.input_bar,
            text="📸 Teach",
            width=70,
            corner_radius=12,
            command=self._on_teach_clicked
        )
        self.teach_button.grid(row=0, column=1, padx=5, pady=5)

        # send button
        self.send_button = ctk.CTkButton(
            self.input_bar,
            text="Send ➤",
            width=70,
            corner_radius=12,
            command=self._on_send_clicked
        )
        self.send_button.grid(row=0, column=2, padx=5, pady=5)

        # quick ask buttons
        self.q1_button = ctk.CTkButton(
            self.input_bar,
            text="Looking?",
            width=110,
            corner_radius=12,
            command=lambda: self._set_quick_text("what are you looking at?")
        )
        self.q1_button.grid(row=0, column=3, padx=3, pady=5)

        self.q2_button = ctk.CTkButton(
            self.input_bar,
            text="Know?",
            width=100,
            corner_radius=12,
            command=lambda: self._set_quick_text("do you know this?")
        )
        self.q2_button.grid(row=0, column=4, padx=3, pady=5)

        self.q3_button = ctk.CTkButton(
            self.input_bar,
            text="Who am I?",
            width=110,
            corner_radius=12,
            command=lambda: self._set_quick_text("who am I?")
        )
        self.q3_button.grid(row=0, column=5, padx=3, pady=5)

        # return/enter triggers send
        self.root.bind("<Return>", lambda e: self._on_send_clicked())

    # ------------------------------------------------------------
    # CHAT MESSAGES
    # ------------------------------------------------------------
    def _add_message(self, who, text):
        """
        who: "baby" or "me"
        text: string
        We render:
            [avatar] [bubble]
        If from baby:
            add 👍 👎 row below bubble for RL feedback
        """
        turn_outer = ctk.CTkFrame(self.chat_area, fg_color="transparent")

        row = ctk.CTkFrame(turn_outer, fg_color="transparent")

        if who == "baby":
            bubble_color = BOT_BUBBLE_COLOR
            text_color = BOT_TEXT_COLOR
            avatar_img = self.avatar_bot
            anchor_side = "w"
        else:
            bubble_color = USER_BUBBLE_COLOR
            text_color = USER_TEXT_COLOR
            avatar_img = self.avatar_user
            anchor_side = "e"

        bubble = ctk.CTkFrame(
            row,
            fg_color=bubble_color,
            corner_radius=BUBBLE_RADIUS
        )

        ctk.CTkLabel(
            bubble,
            text=text,
            text_color=text_color,
            wraplength=200,
            justify="left"
        ).pack(padx=20, pady=5)

        if who == "baby":
            if avatar_img:
                ctk.CTkLabel(row, image=avatar_img, text="").pack(
                    side="left", padx=5, pady=5
                )
            bubble.pack(side="left", padx=5, pady=5)
        else:
            bubble.pack(side="right", padx=5, pady=5)
            if avatar_img:
                ctk.CTkLabel(row, image=avatar_img, text="").pack(
                    side="right", padx=5, pady=5
                )

        row.pack(anchor=anchor_side, pady=(4, 0), padx=10)

        if who == "baby":
            self.last_baby_reply = text

            fb_row = ctk.CTkFrame(turn_outer, fg_color="transparent")

            good_btn = ctk.CTkButton(
                fb_row,
                text="👍",
                width=50,
                height=32,
                corner_radius=12,
                command=lambda: self._thumb_feedback(True)
            )
            good_btn.pack(side="left", padx=5, pady=(2, 6))

            bad_btn = ctk.CTkButton(
                fb_row,
                text="👎",
                width=50,
                height=32,
                corner_radius=12,
                command=lambda: self._thumb_feedback(False)
            )
            bad_btn.pack(side="left", padx=5, pady=(2, 6))

            fb_row.pack(anchor=anchor_side, padx=65)

        turn_outer.pack(anchor=anchor_side, fill="x")

        self._scroll_chat_to_bottom()

    def _thumb_feedback(self, is_good: bool):
        """Send 👍/👎 to reward_last_reply()"""
        try:
            reward_last_reply(True if is_good else False, rl_agent=self.rl)
            print("✅ RL feedback recorded:", "GOOD" if is_good else "BAD")
        except Exception as e:
            print("⚠️ RL feedback failed:", e)

    def _scroll_chat_to_bottom(self):
        """always scroll chat to bottom after new msg"""
        try:
            for attr in ("_parent_canvas", "_canvas", "canvas"):
                canvas = getattr(self.chat_area, attr, None)
                if canvas:
                    try:
                        canvas.update_idletasks()
                        canvas.yview_moveto(1.0)
                        return
                    except Exception:
                        pass
            self.chat_area.update_idletasks()
        except Exception:
            pass

    def _set_quick_text(self, text: str):
        """preset one of the helper questions in the entry box"""
        try:
            self.entry.delete(0, "end")
            self.entry.insert(0, text)
            self.entry.focus()
        except Exception:
            pass

    # ------------------------------------------------------------
    # BUTTON CALLBACKS
    # ------------------------------------------------------------
    def _on_send_clicked(self, preset_msg=None):
        """called by Send button / Enter key"""
        if preset_msg is not None:
            user_msg = preset_msg.strip()
        else:
            user_msg = self.entry.get().strip()

        if not user_msg:
            return

        if preset_msg is None:
            self.entry.delete(0, "end")

        # show user bubble
        self._add_message("me", user_msg)

        # get baby's reply (passes RL + memory + vision)
        vs = self.current_vision_state
        baby_msg = generate_reply(user_msg, vs, self.memory, self.rl)

        # remember context for RL thumbs
        state_key = "chat"  # simple, or you can make it f"chat|{vs.get('focus_type','none')}"
        set_last_context(state_key, baby_msg)

        # show baby bubble (with feedback buttons)
        self._add_message("baby", baby_msg)

    def _on_teach_clicked(self):
        """
        Teach flow:
        - Require focus on an object
        - Ask user for its name
        - Save in memory + visual feature store
        - Give RL reward for the mapping
        """
        vs = self.current_vision_state

        if vs is None or vs.get("focus_type") != "object":
            messagebox.showinfo("Teach", "I can't see an object right now.")
            return

        base_label = vs.get("focus_label", "this object")
        crop_img = vs.get("current_crop", None)
        conf = vs.get("confidence", 0.0)

        taught_name = simpledialog.askstring(
            "Teach Object",
            f"What is this? (I think '{base_label}', conf {conf:.2f})"
        )

        if not taught_name:
            return

        # --- 1️⃣ Save text memory ---
        self.memory.teach(base_label, taught_name)

        # --- 2️⃣ Save visual crop (feature memory) ---
        try:
            if crop_img is not None:
                obj_id = remember_object(base_label, taught_name, crop_img)
                print(
                    f"[DEBUG] Saved visual crop → ID {obj_id}, "
                    f"Label '{base_label}', Name '{taught_name}'"
                )
            else:
                print("⚠️ No crop found in vision_state to save.")
        except Exception as e:
            print("⚠️ Could not save visual object:", e)

        # --- 3️⃣ Reward RL ---
        try:
            self.rl.record_outcome(
         state_key=f"teaching|{base_label}",
         action=f"remembered {taught_name}",
         reward=+1,
            )
        except Exception as e:
            print("⚠️ RL record_outcome failed:", e)

        # --- 4️⃣ Baby reply ---
        self._add_message("baby", f"Okay! I'll remember this is {taught_name}.")

    # ------------------------------------------------------------
    # BABY FACE + VISION LOOP
    # ------------------------------------------------------------
    def _load_baby_images(self):
        for mood, path in BABY_IMAGE_PATHS.items():
            self.baby_images[mood] = make_square_image(path, BABY_DISPLAY_SIZE)
        self._set_baby_image("idle")

    def _set_baby_image(self, mood_key):
        img = self.baby_images.get(mood_key)
        if img:
            self.baby_label.configure(image=img)

    def _decide_baby_image_key(self, vs):
        """
        - happy if object is known (taught)
        - thinking if object is unknown
        - look left/right/center if face
        - idle otherwise
        """
        if vs is None:
            return "idle"

        focus_type = vs.get("focus_type")
        focus_side = vs.get("focus_side")
        focus_label = vs.get("focus_label")

        if focus_type == "object":
            if focus_label and self.memory.recall(focus_label):
                return "happy"
            return "thinking"

        if focus_type == "face":
            if focus_side in ("left", "right"):
                return focus_side
            return "center"

        return "idle"

    def _vision_loop(self):
        """background thread: poll camera -> update current_vision_state"""
        while self._vision_thread_running:
            vs, _frame = self.vision.step()
            self.current_vision_state = vs
            time.sleep(0.05)

        self.vision.release()

    def _schedule_gui_refresh(self):
        # 1. update baby sprite mood
        mood = self._decide_baby_image_key(self.current_vision_state)
        self._set_baby_image(mood)

        # 2. update left Memory panel live
        if hasattr(self, "memory_text"):
            vs = self.current_vision_state
            if vs:
                raw_label = vs.get("focus_label", "none")
                side = vs.get("focus_side", "-")
                conf = vs.get("confidence", 0.0)
                taught_name = self.memory.recall(raw_label)

                if taught_name:
                    desc = (
                        f"Seeing: {raw_label}\n"
                        f"Known as: {taught_name}\n"
                        f"Side: {side}\n"
                        f"Conf: {conf:.2f}"
                    )
                else:
                    desc = (
                        f"Seeing: {raw_label}\n"
                        f"Unknown object\n"
                        f"Side: {side}\n"
                        f"Conf: {conf:.2f}"
                    )
            else:
                desc = "No object detected."
            self.memory_text.configure(text=desc)

        # resched the periodic update
        self.root.after(100, self._schedule_gui_refresh)

    # ------------------------------------------------------------
    # CLEAN SHUTDOWN
    # ------------------------------------------------------------
    def shutdown(self):
        self._vision_thread_running = False
        time.sleep(0.1)
        self.root.destroy()


# ---------- MAIN ----------
if __name__ == "__main__":
    app = ctk.CTk()
    gui = BabyXApp(app)
    app.protocol("WM_DELETE_WINDOW", gui.shutdown)
    app.mainloop()
