#!/usr/bin/env python3
"""
baby_brain.py
-----------------
This is BabyX's "mind" / reply logic.

It connects:
- vision_state          (what BabyVision sees right now)
- memory (BabyMemory)   (manual teaching: bottle -> "Baby's bottle")
- rl_agent (RLAgent)    (reinforcement: which replies get rewarded)
- object_store          (instance-level ORB features & crops)

Main functions exposed:
- describe_what_baby_sees(vision_state, memory, rl_agent)
- generate_reply(user_msg, vision_state, memory, rl_agent=None)
- set_last_context(state_key, reply)
- reward_last_reply(is_good: bool, rl_agent=None)   (called by 👍 / 👎 buttons)
"""

from typing import Optional, Tuple
from object_store import identify_object

# ============================================================
# RL CONTEXT FOR THUMBS (👍 / 👎)
# ============================================================

# We remember the last (state_key, reply) so that when you click 👍 / 👎,
# we know *what* to give reward to in rl_policy.json.
_last_context = {
    "state_key": None,
    "reply": None,
}


def set_last_context(state_key: str, reply: str):
    """
    Store the last interaction context so RL can learn from feedback.

    state_key: a string describing the situation, e.g.
               "see|object cell phone", "greeting|face", etc.
    reply:     the exact text Baby sent to the user
    """
    _last_context["state_key"] = state_key
    _last_context["reply"] = reply


def get_last_context():
    """Return the state and reply most recently produced by generate_reply."""
    return _last_context["state_key"], _last_context["reply"]


def reward_last_reply(is_good: bool, rl_agent=None):
    """
    Called from GUI when you press 👍 / 👎.

    If rl_agent is provided, we:
      - look up the last (state_key, reply)
      - call rl_agent.record_outcome(...) with +1 / -1
      - rl_agent writes to rl_policy.json

    If rl_agent is None, we just log and do nothing.
    """
    reward = 1.0 if is_good else -1.0

    try:
        if (
            rl_agent is not None
            and _last_context["state_key"]
            and _last_context["reply"]
        ):
            rl_agent.record_outcome(
                state_key=_last_context["state_key"],
                action=_last_context["reply"],
                reward=reward,
            )
            print(
                f"[baby_brain] RL feedback stored → "
                f"state={_last_context['state_key']!r}, "
                f"reply={_last_context['reply']!r}, reward={reward}"
            )
        else:
            print(
                "[baby_brain] RL feedback: no rl_agent or no context; "
                "nothing saved."
            )
    except Exception as e:
        print("[baby_brain] RL feedback failed:", e)


# ============================================================
# helper: pick best known name for an object label
# ============================================================

def get_best_known_name(
    base_label: str,
    memory,
    rl_agent=None,
) -> Optional[str]:
    """
    base_label = raw detector label, like "phone" or "bottle"

    We try to decide what to call that object in words Baby should say.

    For now we keep it simple:
    1. Memory mapping from baby_memory.json
    2. None if unknown

    (We *don't* use rl_agent here anymore, since your RLAgent is
     focused on dialog replies, not label -> name mapping.)
    """
    if not base_label:
        return None

    # 1. ask normal memory
    try:
        taught_name = memory.recall(base_label)
    except Exception as e:
        print("[baby_brain] memory.recall failed:", e)
        taught_name = None

    if taught_name:
        return taught_name

    # 2. nothing
    return None


# ============================================================
# helper: try instance-level match from object_store
# ============================================================

def _try_instance_match(
    vision_state,
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Use ORB features from object_store if we have a crop.

    Returns:
        (best_name, best_label, score)

    - best_name: taught name like "Keshav's phone"
    - best_label: base YOLO label like "cell phone"
    - score: number of feature matches (higher is better)
    """
    if vision_state is None:
        return (None, None, 0)

    crop = vision_state.get("current_crop")
    if crop is None:
        return (None, None, 0)

    try:
        best_name, best_label, score = identify_object(crop)
        if best_name:
            print(
                f"[baby_brain] instance match → {best_name} "
                f"({best_label}), score={score}"
            )
        else:
            print("[baby_brain] instance match → none")
        return (best_name, best_label, score)
    except Exception as e:
        print("[baby_brain] identify_object failed:", e)
        return (None, None, 0)


# ============================================================
# describe_what_baby_sees
# ============================================================

def describe_what_baby_sees(
    vision_state,
    memory,
    rl_agent=None,
) -> str:
    """
    Turn the current vision info into baby-friendly text.

    vision_state is expected to look like:
    {
        "focus_type": "object" / "face" / "none",
        "focus_label": "phone",
        "focus_side": "left" / "center" / "right",
        "current_crop": np.array or None,
        ...
    }
    """

    if vision_state is None:
        return "I don't see anything right now."

    focus_type = vision_state.get("focus_type")
    focus_label = vision_state.get("focus_label", "something")
    focus_side = vision_state.get("focus_side", "center")

    # 1. Baby is looking at an object
    if focus_type == "object":
        # 1a. Try instance-level match first (ORB features)
        inst_name, inst_label, inst_score = _try_instance_match(vision_state)
        INSTANCE_THRESHOLD = 15  # tune this later if needed

        if inst_name and inst_score >= INSTANCE_THRESHOLD:
            nice_name = inst_name
        else:
            # 1b. Fall back to label-level name (memory)
            nice_name = get_best_known_name(focus_label, memory, rl_agent)

        if nice_name:
            # baby knows this thing
            if focus_side == "left":
                side_phrase = "on my left"
            elif focus_side == "right":
                side_phrase = "on my right"
            else:
                side_phrase = "in front of me"

            return f"I see {nice_name} {side_phrase}."
        else:
            # baby doesn't know it yet
            return (
                "I'm looking at something, but I don't know what it is yet. "
                "You can teach me with the camera and the Teach button."
            )

    # 2. Baby is looking at a face
    if focus_type == "face":
        if focus_side == "left":
            return "I see a face on my left."
        elif focus_side == "right":
            return "I see a face on my right."
        else:
            return "I see your face. Hi!"

    # 3. Nothing in particular
    return "I'm not looking at anything special right now."


# ============================================================
# tiny intent detectors
# ============================================================

def _is_ask_what_you_see(msg: str) -> bool:
    msg_low = msg.lower().strip()
    return (
        "what are you looking at" in msg_low
        or "what r u looking at" in msg_low
        or "what are u looking at" in msg_low
        or "what do you see" in msg_low
        or "what u see" in msg_low
        or "what are you seeing" in msg_low
        or "looking?" in msg_low
    )


def _is_ask_do_you_know_this(msg: str) -> bool:
    msg_low = msg.lower().strip()
    return (
        "do you know this" in msg_low
        or "do u know this" in msg_low
        or "do you know what this is" in msg_low
        or "do u know what this is" in msg_low
        or "know?" in msg_low
    )


def _is_ask_who_am_i(msg: str) -> bool:
    msg_low = msg.lower().strip()
    return (
        "who am i" in msg_low
        or "who am i?" in msg_low
        or "who am i ?" in msg_low
        or "who am i baby" in msg_low
        or "who am i baby?" in msg_low
        or "who am i to you" in msg_low
        or "whats my name" in msg_low
        or "what's my name" in msg_low
        or "whoami" in msg_low
    )


def _is_greeting(msg: str) -> bool:
    msg_low = msg.lower().strip()
    return any(
        g in msg_low
        for g in (
            "hi", "hii", "hiii", "hello", "hey", "heyy", "yo", "sup",
            "good morning", "good night", "good afternoon", "good evening",
            "baby", "babyy", "babyx",
        )
    )


def _is_love_msg(msg: str) -> bool:
    msg_low = msg.lower().strip()
    return (
        "love you" in msg_low
        or "i love u" in msg_low
        or "i love you" in msg_low
        or "luv u" in msg_low
        or "luv you" in msg_low
        or "cute" in msg_low
        or "ur cute" in msg_low
        or "you're cute" in msg_low
    )


# ============================================================
# generate_reply (main brain entrypoint)
# ============================================================

def generate_reply(
    user_msg: str,
    vision_state,
    memory,
    rl_agent=None,
) -> str:
    """
    Decide what Baby should say back in chat.

    rl_agent can be None (older code), but if you pass in an RLAgent,
    Baby will use it to talk about known objects by their best name,
    and we'll also log context for thumbs-up/down RL.

    NOTE: RL is *not yet* changing the reply selection here.
          It is just logging feedback into rl_policy.json.
    """

    msg_low = user_msg.lower().strip()

    # Helper to set context before we return:
    def _finish(reply_text: str, state_key: str):
        # remember for thumbs feedback
        set_last_context(state_key=state_key, reply=reply_text)
        return reply_text

    # 1. user is basically asking what baby sees right now
    if _is_ask_what_you_see(msg_low):
        reply = describe_what_baby_sees(vision_state, memory, rl_agent)
        focus_type = vision_state.get("focus_type") if vision_state else "none"
        state = f"see|{focus_type}"
        return _finish(reply, state)

    # 2. "do you know this?" → say if we recognize the focused object
    if _is_ask_do_you_know_this(msg_low):
        if vision_state is None:
            return _finish("I can't see anything right now.", "know|none")

        if vision_state.get("focus_type") == "object":
            raw_label = vision_state.get("focus_label", "this")

            # instance-level first
            inst_name, inst_label, inst_score = _try_instance_match(vision_state)
            INSTANCE_THRESHOLD = 15

            if inst_name and inst_score >= INSTANCE_THRESHOLD:
                reply = f"Yes, I think that's {inst_name}."
                return _finish(reply, f"know|object {raw_label}")

            # else fallback to label-level
            nice_name = get_best_known_name(raw_label, memory, rl_agent)
            if nice_name:
                reply = f"Yes, I think that's {nice_name}."
                return _finish(reply, f"know|object {raw_label}")
            else:
                reply = "Not yet. You can teach me with the Teach button."
                return _finish(reply, f"know|object {raw_label}")
        else:
            return _finish(
                "I'm not looking at an object right now.",
                f"know|{vision_state.get('focus_type', 'none')}",
            )

    # 3. "who am I?"
    if _is_ask_who_am_i(msg_low):
        reply = "You're my human. 🥺💖"
        return _finish(reply, "whoami")

    # 4. greeting / filler vibe
    if _is_greeting(msg_low):
        if vision_state and vision_state.get("focus_type") == "object":
            raw_label = vision_state.get("focus_label", "something")

            inst_name, inst_label, inst_score = _try_instance_match(vision_state)
            INSTANCE_THRESHOLD = 15

            if inst_name and inst_score >= INSTANCE_THRESHOLD:
                reply = f"Hii 👋 I'm looking at {inst_name} right now."
                return _finish(reply, f"greeting|object {raw_label}")

            nice_name = get_best_known_name(raw_label, memory, rl_agent)
            if nice_name:
                reply = f"Hii 👋 I'm looking at {nice_name} right now."
                return _finish(reply, f"greeting|object {raw_label}")
            else:
                reply = "Hii 👋 I'm still trying to learn what I'm seeing."
                return _finish(reply, f"greeting|object {raw_label}")
        else:
            reply = "Hii 👋 I'm here."
            focus_type = vision_state.get("focus_type") if vision_state else "none"
            return _finish(reply, f"greeting|{focus_type}")

    # 5. affection
    if _is_love_msg(msg_low):
        reply = "🥺💖 I love you too."
        return _finish(reply, "love")

    # 6. fallback: also give a little status so it feels alive
    if vision_state is not None:
        focus_type = vision_state.get("focus_type")
        if focus_type == "object":
            raw_label = vision_state.get("focus_label", "something")

            inst_name, inst_label, inst_score = _try_instance_match(vision_state)
            INSTANCE_THRESHOLD = 15

            if inst_name and inst_score >= INSTANCE_THRESHOLD:
                reply = f"I'm learning. I think I see {inst_name}."
                return _finish(reply, f"fallback|object {raw_label}")
            else:
                nice_name = get_best_known_name(raw_label, memory, rl_agent)
                if nice_name:
                    reply = f"I'm learning. I think I see {nice_name}."
                    return _finish(reply, f"fallback|object {raw_label}")
                else:
                    reply = "I'm learning. I see something but I don't know it yet."
                    return _finish(reply, f"fallback|object {raw_label}")
        elif focus_type == "face":
            reply = "I'm looking at you 👀"
            return _finish(reply, "fallback|face")
        else:
            reply = "I'm here and listening."
            return _finish(reply, f"fallback|{focus_type}")

    # totally nothing known about vision → super fallback
    reply = "I am learning."
    return _finish(reply, "fallback|none")
