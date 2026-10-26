// Input Buffer — predictive input buffering for Geometry Dash 2.2081 (Geode v5)
//
// How it works
// ------------
// Every jump-button input already flows through GJBaseGameLayer::handleButton,
// which GD calls from inside its physics step after resolving queued inputs.
// We intercept that call, decide whether the input should pass straight through,
// be delayed, or have its release held back, and keep a tiny per-player schedule
// that is flushed at the start of each physics step (processCommands).
//
//  * Ground / orb buffer (cube, ball, robot, spider)
//      A tap made in mid-air whose release arrives before it has "done" anything
//      gets its release postponed for N ticks. While GD still sees the button
//      held, its own hold logic jumps on landing / activates the orb on the exact
//      collision tick. As soon as a jump, flip, orb or dash orb is used, the
//      release is replayed with the player's original hold length (so robot
//      jump height is preserved). If GD's native hold logic does not fire for a
//      gamemode, a fallback re-click is issued on the tick after contact.
//
//  * Tap stretching (ship, UFO, swing)
//      Releases are pushed back so a press is held for at least N ticks.
//
//  * Wave: Min Hold / Min Gap / Grid Snap (see Config below).
//
//  * Smart Assist (experimental, Windows only, opt-in)
//      On every press/release, a hidden copy of the player is simulated forward
//      with the input applied now, 1 tick later, 2 ticks later... If applying it
//      now leads to a crash within the lookahead but a slightly later timing
//      survives, the input is delayed to the earliest surviving tick. It can only
//      make inputs later, never earlier. Simulation approach is modelled on the
//      open-source xdBot "Show Trajectory" (fake PlayerObject + checkCollisions).
//
// The real player is never moved or edited. The game only ever sees ordinary
// press/release events, just at slightly adjusted ticks.

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/HardStreak.hpp>
#include <Geode/modify/GameObject.hpp>
#include <Geode/modify/EffectGameObject.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "copy_state.hpp"

using namespace geode::prelude;

namespace ib {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

constexpr double kVanillaTps = 240.0;

enum class WaveMode { None, MinHold, MinGap, GridSnap };

struct Config {
    bool enabled = true;
    bool verbose = true;
    bool groundOrb = true;
    bool tapStretch = true;
    bool assist = false;
    WaveMode wave = WaveMode::None;
    int64_t targetTps = 240;

    // All values below are already scaled to targetTps.
    int bufferTicks = 4;
    int stretchTicks = 2;
    int waveHoldTicks = 2;
    int waveGapTicks = 2;
    int waveGridTicks = 2;
    int assistMaxShift = 4;
    int assistLookahead = 36;

    void refresh() {
        auto mod = Mod::get();
        enabled = mod->getSettingValue<bool>("enabled");
        verbose = mod->getSettingValue<bool>("debug-logging");
        groundOrb = mod->getSettingValue<bool>("ground-orb-buffer");
        tapStretch = mod->getSettingValue<bool>("tap-stretch");
        assist = mod->getSettingValue<bool>("smart-assist");
        targetTps = std::clamp<int64_t>(mod->getSettingValue<int64_t>("target-tps"), 60, 2000);

        // Settings are authored in 240-TPS ticks; scale so the real-time window
        // stays constant when a TPS bypass changes the step rate.
        auto scaled = [&](char const* key) {
            double raw = static_cast<double>(mod->getSettingValue<int64_t>(key));
            return std::max(1, static_cast<int>(std::lround(raw * targetTps / kVanillaTps)));
        };
        bufferTicks = scaled("buffer-ticks");
        stretchTicks = scaled("tap-stretch-ticks");
        waveHoldTicks = scaled("wave-min-hold-ticks");
        waveGapTicks = scaled("wave-min-gap-ticks");
        waveGridTicks = scaled("wave-grid-ticks");
        assistMaxShift = scaled("assist-max-shift");
        assistLookahead = scaled("assist-lookahead");

        auto w = mod->getSettingValue<std::string>("wave-mode");
        if (w == "Min Hold") wave = WaveMode::MinHold;
        else if (w == "Min Gap") wave = WaveMode::MinGap;
        else if (w == "Grid Snap") wave = WaveMode::GridSnap;
        else wave = WaveMode::None;
    }
};

Config g_cfg;

#define IB_LOG(...)                                      \
    do {                                                 \
        if (::ib::g_cfg.verbose) log::info(__VA_ARGS__); \
    } while (0)

// ---------------------------------------------------------------------------
// Simulation state (Smart Assist)
// ---------------------------------------------------------------------------

struct Sim {
    PlayerObject* fake = nullptr; // hidden player used only for prediction
    PlayLayer* owner = nullptr;   // the PlayLayer the fake belongs to
    bool active = false;          // true while a prediction is running
    bool died = false;            // set by the destroyPlayer hook during a prediction
    float dt = 0.25f;             // last real physics step dt (captured from PlayerObject::update)
};

Sim g_sim;

inline bool simulating(PlayerObject* p = nullptr) {
    return g_sim.active || (p && p == g_sim.fake);
}

// Object types/IDs that matter for survival, everything else is switched off
// during a prediction (same lists as xdBot's trajectory).
const std::unordered_set<int> kPortalIDs = { 101, 99, 11, 10, 200, 201, 202, 203, 1334 };
const std::unordered_set<int> kSolidTypes = { 0, 2, 47, 25 };

inline void applySimPortal(PlayerObject* player, int id) {
    switch (id) {
        case 101: player->togglePlayerScale(true, true); player->updatePlayerScale(); break;
        case 99: player->togglePlayerScale(false, true); player->updatePlayerScale(); break;
        case 200: player->m_playerSpeed = 0.7f; break;
        case 201: player->m_playerSpeed = 0.9f; break;
        case 202: player->m_playerSpeed = 1.1f; break;
        case 203: player->m_playerSpeed = 1.3f; break;
        case 1334: player->m_playerSpeed = 1.6f; break;
        default: break;
    }
}

inline void clearLog(cocos2d::CCDictionary* d) {
    if (d) d->removeAllObjects();
}

// ---------------------------------------------------------------------------
// Per-player state
// ---------------------------------------------------------------------------

enum class Category { Buffer, Stretch, Wave };

inline Category categorize(PlayerObject* p) {
    if (p->m_isDart) return Category::Wave;
    if (p->m_isShip || p->m_isBird || p->m_isSwing) return Category::Stretch; // m_isBird == UFO
    return Category::Buffer; // cube, ball, robot, spider
}

inline char const* categoryName(Category c) {
    switch (c) {
        case Category::Buffer: return "ground/orb";
        case Category::Stretch: return "stretch";
        case Category::Wave: return "wave";
    }
    return "?";
}

// Why a release is currently being held back.
enum class Defer { None, Buffer, Replay, Stretch, WaveHold, WaveGrid, Refire, Delayed, Assist };

inline uint64_t ceilToGrid(uint64_t t, int grid) {
    uint64_t g = static_cast<uint64_t>(std::max(1, grid));
    return ((t + g - 1) / g) * g;
}

struct SlotState {
    bool physDown = false;        // what the player's finger is doing
    bool gameDown = false;        // what we last told the game
    uint64_t realPressTick = 0;   // tick of the last physical press
    uint64_t gamePressTick = 0;   // tick the game last saw a press
    uint64_t lastGameRelease = 0; // tick the game last saw a release
    bool hasReleased = false;

    // Ground / orb buffer
    bool airPress = false;        // armed: pressed while airborne
    bool consumed = false;        // the press already caused a jump/orb/flip
    uint64_t bufferDeadline = 0;
    int heldTicks = 1;            // physical hold length, replayed after contact

    // Schedule (at most one pending press and one pending release)
    std::optional<uint64_t> pendingPress;
    std::optional<uint64_t> pendingRelease;
    int holdAfterPress = -1;      // >= 0: player already let go during a delayed press
    Defer deferReason = Defer::None;
    bool refireQueued = false;
};

} // namespace ib

using namespace ib;

// ---------------------------------------------------------------------------
// GJBaseGameLayer: input interception + per-tick scheduler + prediction
// (buffering covers PlayLayer and editor playtests; Smart Assist is PlayLayer only)
// ---------------------------------------------------------------------------

class $modify(IBGameLayer, GJBaseGameLayer) {
public:
    struct Fields {
        std::array<SlotState, 2> m_slots{};
        uint64_t m_tick = 0;
        unsigned int m_lastProgress = 0;
        bool m_injecting = false;
    };

    // --- helpers -----------------------------------------------------------

    PlayerObject* slotPlayer(int slot) {
        return slot == 0 ? m_player1 : m_player2;
    }

    bool twoPlayer() {
        return m_levelSettings && m_levelSettings->m_twoPlayerMode;
    }

    int slotOf(PlayerObject* p) {
        if (p == m_player1) return 0;
        if (p == m_player2 && twoPlayer()) return 1;
        return -1; // player 2 in non-2P dual follows slot 0's inputs; ignore its events
    }

    // Sends an input to the original handleButton (the rest of the hook chain).
    void forward(int slot, bool down) {
        auto f = m_fields.self();
        f->m_injecting = true;
        GJBaseGameLayer::handleButton(down, static_cast<int>(PlayerButton::Jump), slot == 0);
        f->m_injecting = false;
        f->m_slots[slot].gameDown = down;
    }

    void pressNow(int slot, PlayerObject* player, bool armBuffer) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        st.consumed = false; // must be cleared before the game can react
        st.refireQueued = false;
        st.gamePressTick = f->m_tick;
        st.airPress = armBuffer && g_cfg.groundOrb &&
                      categorize(player) == Category::Buffer && !player->m_isOnGround;
        st.bufferDeadline = f->m_tick + static_cast<uint64_t>(g_cfg.bufferTicks);
        forward(slot, true);
    }

    void releaseNow(int slot) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        st.pendingRelease.reset();
        st.deferReason = Defer::None;
        st.airPress = false;
        st.lastGameRelease = f->m_tick;
        st.hasReleased = true;
        forward(slot, false);
    }

    // When should a release that arrives *now* actually reach the game?
    uint64_t releaseTarget(SlotState& st, Category cat, uint64_t now, Defer& why) {
        why = Defer::None;
        switch (cat) {
            case Category::Buffer:
                if (g_cfg.groundOrb && st.airPress && !st.consumed && now < st.bufferDeadline) {
                    why = Defer::Buffer;
                    return st.bufferDeadline;
                }
                break;
            case Category::Stretch:
                if (g_cfg.tapStretch) {
                    uint64_t t = st.gamePressTick + static_cast<uint64_t>(g_cfg.stretchTicks);
                    if (t > now) { why = Defer::Stretch; return t; }
                }
                break;
            case Category::Wave:
                if (g_cfg.wave == WaveMode::MinHold) {
                    uint64_t t = st.gamePressTick + static_cast<uint64_t>(g_cfg.waveHoldTicks);
                    if (t > now) { why = Defer::WaveHold; return t; }
                }
                else if (g_cfg.wave == WaveMode::GridSnap) {
                    uint64_t t = ceilToGrid(now, g_cfg.waveGridTicks);
                    if (t <= st.gamePressTick) t = st.gamePressTick + static_cast<uint64_t>(g_cfg.waveGridTicks);
                    if (t > now) { why = Defer::WaveGrid; return t; }
                }
                break;
        }
        return now;
    }

    // --- Smart Assist ------------------------------------------------------

    // Plays the hidden player forward from the real player's current state.
    // The input is applied `delay` ticks from now (and, for a tap, released one
    // tick later). Returns how many ticks it stayed alive, capped at `steps`.
    int simulateLife(PlayerObject* real, bool down, int delay, bool tap, int steps) {
        auto fake = g_sim.fake;
        copyPhysicsState(fake, real);
        fake->setPosition(real->getPosition());
        fake->setRotation(real->getRotation());
        fake->updatePlayerScale();
        g_sim.died = false;

        float dt = g_sim.dt;
        for (int i = 0; i < steps; i++) {
            clearLog(fake->m_collisionLogTop);
            clearLog(fake->m_collisionLogBottom);
            clearLog(fake->m_collisionLogLeft);
            clearLog(fake->m_collisionLogRight);

            this->checkCollisions(fake, dt, false);
            if (g_sim.died) return i;

            if (i == delay) {
                if (down) fake->pushButton(PlayerButton::Jump);
                else fake->releaseButton(PlayerButton::Jump);
            }
            else if (tap && i == delay + 1) {
                fake->releaseButton(PlayerButton::Jump);
            }

            fake->update(dt);
            fake->updateRotation(dt);
            fake->updatePlayerScale();
        }
        return steps;
    }

    // How many ticks to delay this input (0 = leave it alone).
    int assistDelay(int slot, PlayerObject* player, bool down) {
#ifndef GEODE_IS_WINDOWS
        (void)slot; (void)player; (void)down;
        return 0;
#else
        if (!g_cfg.assist || !g_sim.fake || g_sim.active) return 0;
        auto pl = PlayLayer::get();
        if (!pl || g_sim.owner != pl || static_cast<GJBaseGameLayer*>(pl) != this) return 0;
        if (player->m_isDead) return 0;

        bool tap = down && categorize(player) == Category::Buffer;
        int look = g_cfg.assistLookahead;

        g_sim.active = true;
        int life0 = simulateLife(player, down, 0, tap, look);
        int chosen = 0;
        int chosenLife = life0;
        if (life0 < look) {
            for (int d = 1; d <= g_cfg.assistMaxShift; d++) {
                int life = simulateLife(player, down, d, tap, look + d);
                if (life >= look + d) { chosen = d; chosenLife = life; break; }
            }
        }
        g_sim.active = false;
        g_sim.died = false;

        if (chosen > 0) {
            IB_LOG("[InputBuffer] P{} ASSIST: {} shifted +{} ticks (on time: crash in {} ticks, shifted: clear for {})",
                slot + 1, down ? "press" : "release", chosen, life0, chosenLife);
        }
        return chosen;
#endif
    }

    // --- physical input ----------------------------------------------------

    void onRealPress(int slot, PlayerObject* player) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        auto now = f->m_tick;
        auto cat = categorize(player);

        st.physDown = true;
        st.realPressTick = now;

        // A delayed press is still waiting; the player is simply holding again.
        if (st.pendingPress) {
            st.holdAfterPress = -1;
            return;
        }

        // The game is still holding from a postponed release.
        if (st.pendingRelease) {
            if (cat == Category::Buffer) {
                // A fresh click supersedes the buffered one: real release + press.
                IB_LOG("[InputBuffer] P{} new click at tick {} replaces buffered tap", slot + 1, now);
                releaseNow(slot);
            }
            else {
                // Holding modes: just keep holding, merge into one continuous press.
                st.pendingRelease.reset();
                st.deferReason = Defer::None;
                IB_LOG("[InputBuffer] P{} press at tick {} merged into stretched hold", slot + 1, now);
                return;
            }
        }

        if (st.gameDown) releaseNow(slot); // desync safety: never send two presses in a row

        if (cat == Category::Wave) {
            if (g_cfg.wave == WaveMode::MinGap && st.hasReleased) {
                uint64_t earliest = st.lastGameRelease + static_cast<uint64_t>(g_cfg.waveGapTicks);
                if (now < earliest) {
                    st.pendingPress = earliest;
                    IB_LOG("[InputBuffer] P{} wave press SAVED at tick {}, delayed to {} (min gap)", slot + 1, now, earliest);
                    return;
                }
            }
            else if (g_cfg.wave == WaveMode::GridSnap) {
                uint64_t snapped = ceilToGrid(now, g_cfg.waveGridTicks);
                if (snapped > now) {
                    st.pendingPress = snapped;
                    IB_LOG("[InputBuffer] P{} wave press SAVED at tick {}, snapped to {}", slot + 1, now, snapped);
                    return;
                }
            }
        }

        if (int d = assistDelay(slot, player, true); d > 0) {
            st.pendingPress = now + static_cast<uint64_t>(d);
            return;
        }

        pressNow(slot, player, true);
        if (st.airPress) {
            IB_LOG("[InputBuffer] P{} air press at tick {} armed (window until tick {})",
                slot + 1, now, st.bufferDeadline);
        }
    }

    void onRealRelease(int slot, PlayerObject* player) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        auto now = f->m_tick;
        auto cat = categorize(player);

        st.physDown = false;
        st.heldTicks = std::max<int>(1, static_cast<int>(now - st.realPressTick));

        // Press hasn't reached the game yet: remember how long they held it.
        if (st.pendingPress) {
            st.holdAfterPress = st.heldTicks;
            IB_LOG("[InputBuffer] P{} release at tick {} queued behind delayed press ({} ticks)",
                slot + 1, now, st.heldTicks);
            return;
        }

        if (!st.gameDown) { // nothing held on the game side; just stay in sync
            releaseNow(slot);
            return;
        }

        Defer why;
        uint64_t target = releaseTarget(st, cat, now, why);
        if (target > now) {
            st.pendingRelease = target;
            st.deferReason = why;
            if (why == Defer::Buffer) {
                IB_LOG("[InputBuffer] P{} early tap SAVED at tick {} (held {}), kept alive until tick {}",
                    slot + 1, now, st.heldTicks, target);
            }
            else {
                IB_LOG("[InputBuffer] P{} {} release SAVED at tick {}, delayed to {}",
                    slot + 1, categoryName(cat), now, target);
            }
            return;
        }

        if (int d = assistDelay(slot, player, false); d > 0) {
            st.pendingRelease = now + static_cast<uint64_t>(d);
            st.deferReason = Defer::Assist;
            return;
        }

        releaseNow(slot);
    }

    // --- scheduler ---------------------------------------------------------

    void flushSlot(int slot) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        auto now = f->m_tick;
        auto player = slotPlayer(slot);
        if (!player) return;

        if (st.pendingPress && *st.pendingPress <= now) {
            st.pendingPress.reset();
            pressNow(slot, player, true);
            IB_LOG("[InputBuffer] P{} delayed press fired at tick {}", slot + 1, now);

            if (st.holdAfterPress >= 0) {
                uint64_t r = now + static_cast<uint64_t>(std::max(1, st.holdAfterPress));
                if (categorize(player) == Category::Wave) {
                    if (g_cfg.wave == WaveMode::GridSnap) r = ceilToGrid(r, g_cfg.waveGridTicks);
                    if (g_cfg.wave == WaveMode::MinHold)
                        r = std::max(r, now + static_cast<uint64_t>(g_cfg.waveHoldTicks));
                }
                st.pendingRelease = r;
                st.deferReason = Defer::Delayed;
                st.holdAfterPress = -1;
            }
        }

        if (st.refireQueued) {
            st.refireQueued = false;
            if (st.gameDown) releaseNow(slot);
            pressNow(slot, player, false);
            st.consumed = true; // one fallback per tap, never loop
            if (!st.physDown) {
                st.pendingRelease = now + static_cast<uint64_t>(std::max(1, st.heldTicks));
                st.deferReason = Defer::Refire;
            }
            IB_LOG("[InputBuffer] P{} buffered click re-fired at tick {} (contact fallback)", slot + 1, now);
        }

        if (st.pendingRelease && *st.pendingRelease <= now) {
            releaseNow(slot);
        }
    }

    void postStep(int slot) {
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        auto now = f->m_tick;
        auto player = slotPlayer(slot);
        if (!player) return;

        if (player->m_isDead) {
            bool phys = st.physDown;
            st = SlotState{};
            st.physDown = phys;
            return;
        }

        auto cat = categorize(player);

        // Took a portal while we were holding a buffered tap: let go next tick so a
        // cube tap doesn't turn into a long ship/wave hold.
        if (st.deferReason == Defer::Buffer && st.pendingRelease && cat != Category::Buffer) {
            st.pendingRelease = now + 1;
            st.deferReason = Defer::None;
            st.airPress = false;
            IB_LOG("[InputBuffer] P{} gamemode changed, dropping buffered tap", slot + 1);
            return;
        }

        if (cat != Category::Buffer || !g_cfg.groundOrb || !st.airPress || st.consumed) return;

        if (now > st.bufferDeadline) {
            st.airPress = false;
            IB_LOG("[InputBuffer] P{} buffer expired at tick {} without contact", slot + 1, now);
            return;
        }

        bool onGround = player->m_isOnGround;
        bool inRing = player->m_touchingRings && player->m_touchingRings->count() > 0;
        if (onGround || inRing) {
            // GD's own hold logic didn't use the press this tick; click for it.
            st.refireQueued = true;
        }
    }

    // --- API used by the other hooks ----------------------------------------

    void resetBuffers() {
        auto f = m_fields.self();
        for (auto& st : f->m_slots) st = SlotState{};
        f->m_tick = 0;
        g_cfg.refresh();
    }

    // Called by the PlayerObject hooks whenever a click actually did something.
    void onConsumed(PlayerObject* p, char const* what) {
        int slot = slotOf(p);
        if (slot < 0) return;
        auto f = m_fields.self();
        auto& st = f->m_slots[slot];
        bool wasArmed = st.airPress && !st.consumed;
        st.consumed = true;

        if (st.deferReason == Defer::Buffer && st.pendingRelease) {
            uint64_t r = f->m_tick + static_cast<uint64_t>(std::max(1, st.heldTicks));
            st.pendingRelease = r;
            st.deferReason = Defer::Replay;
            IB_LOG("[InputBuffer] P{} buffered click fired ({}) at tick {}, {} ticks after press; release at {}",
                slot + 1, what, f->m_tick, f->m_tick - st.gamePressTick, r);
        }
        else if (wasArmed && !st.physDown) {
            IB_LOG("[InputBuffer] P{} buffered click fired ({}) at tick {}", slot + 1, what, f->m_tick);
        }
    }

    // --- hooks: input & tick -------------------------------------------------

    void handleButton(bool down, int button, bool isPlayer1) {
        auto f = m_fields.self();
        if (f->m_injecting || g_sim.active || !g_cfg.enabled ||
            button != static_cast<int>(PlayerButton::Jump)) {
            return GJBaseGameLayer::handleButton(down, button, isPlayer1);
        }

        int slot = isPlayer1 ? 0 : 1;
        auto player = slotPlayer(slot);
        if (!player || player->m_isDead) {
            auto& st = f->m_slots[slot];
            st = SlotState{};
            st.physDown = down;
            st.gameDown = down;
            return GJBaseGameLayer::handleButton(down, button, isPlayer1);
        }

        if (down) onRealPress(slot, player);
        else onRealRelease(slot, player);
    }

    // One call == one physics step in 2.2081 (240 per second at vanilla TPS).
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        auto f = m_fields.self();

        // Practice respawns / editor restarts rewind progress: drop stale schedules.
        if (m_gameState.m_currentProgress < f->m_lastProgress) resetBuffers();
        f->m_lastProgress = m_gameState.m_currentProgress;

        f->m_tick++;
        if (f->m_tick % 60 == 0) g_cfg.refresh(); // pick up settings changed mid-level

        // Always flush, even if the mod was just disabled, so nothing stays stuck.
        flushSlot(0);
        flushSlot(1);

        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);

        if (g_cfg.enabled) {
            postStep(0);
            postStep(1);
        }
    }

    // --- hooks: keep predictions side-effect free ----------------------------
    // While a prediction runs, only solid blocks, hazards and size/speed portals
    // exist for the hidden player; orbs, pads, triggers, sounds and collectibles
    // are switched off so the real level is never touched.

    void collisionCheckObjects(PlayerObject* player, gd::vector<GameObject*>* objects, int objectCount, float dt) {
        if (!simulating(player) || !objects) {
            return GJBaseGameLayer::collisionCheckObjects(player, objects, objectCount, dt);
        }

        std::vector<GameObject*> disabled;
        for (auto obj : *objects) {
            if (!obj) continue;
            bool keep = kSolidTypes.contains(static_cast<int>(obj->m_objectType)) ||
                        kPortalIDs.contains(obj->m_objectID);
            if (keep || obj->m_isDisabled || obj->m_isDisabled2) continue;
            obj->m_isDisabled = true;
            obj->m_isDisabled2 = true;
            disabled.push_back(obj);
        }

        GJBaseGameLayer::collisionCheckObjects(player, objects, objectCount, dt);

        for (auto obj : disabled) {
            obj->m_isDisabled = false;
            obj->m_isDisabled2 = false;
        }
    }

    bool canBeActivatedByPlayer(PlayerObject* player, EffectGameObject* object) {
        if (simulating(player)) {
            if (object) applySimPortal(player, object->m_objectID);
            return false;
        }
        return GJBaseGameLayer::canBeActivatedByPlayer(player, object);
    }

    void playerTouchedRing(PlayerObject* player, RingObject* object) {
        if (simulating(player)) return;
        GJBaseGameLayer::playerTouchedRing(player, object);
    }

    void playerTouchedTrigger(PlayerObject* player, EffectGameObject* object) {
        if (simulating(player)) {
            if (object) applySimPortal(player, object->m_objectID);
            return;
        }
        GJBaseGameLayer::playerTouchedTrigger(player, object);
    }

    void activateSFXTrigger(SFXTriggerGameObject* object) {
        if (g_sim.active) return;
        GJBaseGameLayer::activateSFXTrigger(object);
    }

    void activateSongEditTrigger(SongTriggerGameObject* object) {
        if (g_sim.active) return;
        GJBaseGameLayer::activateSongEditTrigger(object);
    }

    void gameEventTriggered(GJGameEvent event, int material, int playerID) {
        if (g_sim.active) return;
        GJBaseGameLayer::gameEventTriggered(event, material, playerID);
    }
};

// ---------------------------------------------------------------------------
// PlayerObject: detect when a press was actually used + prediction guards
// ---------------------------------------------------------------------------

class $modify(IBPlayerObject, PlayerObject) {
public:
    IBGameLayer* ibLayer() {
        return m_gameLayer ? static_cast<IBGameLayer*>(m_gameLayer) : nullptr;
    }

    void update(float dt) {
        PlayerObject::update(dt);
        if (!simulating(this)) g_sim.dt = dt; // remember the real step size
    }

    // Ground jumps, ball/spider flips, robot jumps all bump the jump counter.
    void incrementJumps() {
        if (simulating(this)) return; // never touch stats from a prediction
        PlayerObject::incrementJumps();
        if (auto gl = ibLayer()) gl->onConsumed(this, "jump");
    }

    void ringJump(RingObject* object, bool skipCheck) {
        if (simulating(this)) return;
        PlayerObject::ringJump(object, skipCheck);
        if (auto gl = ibLayer()) gl->onConsumed(this, "orb");
    }

    void startDashing(DashRingObject* object) {
        PlayerObject::startDashing(object);
        if (simulating(this)) return;
        if (auto gl = ibLayer()) gl->onConsumed(this, "dash orb");
    }

    void playSpiderDashEffect(cocos2d::CCPoint from, cocos2d::CCPoint to) {
        if (simulating(this)) return;
        PlayerObject::playSpiderDashEffect(from, to);
    }
};

// ---------------------------------------------------------------------------
// PlayLayer: per-attempt reset, hidden player lifecycle, simulated deaths
// ---------------------------------------------------------------------------

class $modify(IBPlayLayer, PlayLayer) {
public:
    void resetLevel() {
        PlayLayer::resetLevel();
        static_cast<IBGameLayer*>(static_cast<GJBaseGameLayer*>(this))->resetBuffers();
    }

    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();

        // A fake left over from a layer that was torn down without onQuit is
        // abandoned rather than released, since its game layer no longer exists.
        g_sim.fake = nullptr;
        g_sim.owner = nullptr;
        g_sim.active = false;
        g_sim.died = false;

        auto fake = PlayerObject::create(1, 1, this, this, true);
        if (!fake || !m_objectLayer) return;
        fake->retain();
        fake->setPosition({ 0.f, 105.f });
        fake->setVisible(false);
        m_objectLayer->addChild(fake);
        g_sim.fake = fake;
        g_sim.owner = this;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        if (simulating(player)) {
            g_sim.died = true; // a predicted crash, not a real one
            return;
        }
        PlayLayer::destroyPlayer(player, object);
    }

    void playEndAnimationToPos(cocos2d::CCPoint position) {
        if (g_sim.active) return;
        PlayLayer::playEndAnimationToPos(position);
    }

    void onQuit() {
        if (g_sim.owner == this && g_sim.fake) {
            g_sim.fake->removeFromParentAndCleanup(true);
            g_sim.fake->release();
        }
        g_sim.fake = nullptr;
        g_sim.owner = nullptr;
        g_sim.active = false;
        PlayLayer::onQuit();
    }
};

// ---------------------------------------------------------------------------
// Visual side effects to suppress during predictions
// ---------------------------------------------------------------------------

class $modify(IBHardStreak, HardStreak) {
    void addPoint(cocos2d::CCPoint point) {
        if (g_sim.active) return;
        HardStreak::addPoint(point);
    }
};

class $modify(IBGameObject, GameObject) {
    void playShineEffect() {
        if (g_sim.active) return;
        GameObject::playShineEffect();
    }
};

class $modify(IBEffectGameObject, EffectGameObject) {
    void triggerObject(GJBaseGameLayer* layer, int uniqueID, gd::vector<int> const* remapKeys) {
        if (g_sim.active) return;
        EffectGameObject::triggerObject(layer, uniqueID, remapKeys);
    }
};

$on_mod(Loaded) {
    g_cfg.refresh();
    log::info("[InputBuffer] loaded: buffer={}t stretch={}t wave-mode={} assist={} @ {} TPS",
        g_cfg.bufferTicks, g_cfg.stretchTicks,
        Mod::get()->getSettingValue<std::string>("wave-mode"), g_cfg.assist, g_cfg.targetTps);
}
